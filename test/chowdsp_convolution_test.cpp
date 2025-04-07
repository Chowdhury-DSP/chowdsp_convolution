#include <chrono>
#include <iostream>
#include <random>

#include <juce_dsp/juce_dsp.h>

#include <chowdsp_convolution.h>
#include <chowdsp_fft.h>

#include <chowdsp_buffers/chowdsp_buffers.h>
#include <chowdsp_buffers/Buffers/chowdsp_Buffer.cpp> // NOLINT
template class chowdsp::Buffer<float, 32>;
using Convolution_Internal_Buffer = chowdsp::Buffer<float, 32>;

struct ConvolutionEngine
{
    ConvolutionEngine (const float* samples,
                       size_t numSamples,
                       size_t maxBlockSize)
        : blockSize ((size_t) juce::nextPowerOfTwo ((int) maxBlockSize)),
          fftSize (blockSize > 128 ? 2 * blockSize : 4 * blockSize),
          fftObject (std::make_unique<juce::dsp::FFT> (juce::roundToInt (std::log2 (fftSize)))),
          numSegments (numSamples / (fftSize - blockSize) + 1u),
          numInputSegments ((blockSize > 128 ? numSegments : 3 * numSegments)),
          bufferInput (1, static_cast<int> (fftSize)),
          bufferOutput (1, static_cast<int> (fftSize * 2)),
          bufferTempOutput (1, static_cast<int> (fftSize * 2)),
          bufferOverlap (1, static_cast<int> (fftSize))
    {
        bufferOutput.clear();

        auto updateSegmentsIfNecessary = [this] (size_t numSegmentsToUpdate,
                                                 std::vector<Convolution_Internal_Buffer>& segments)
        {
            if (numSegmentsToUpdate == 0
                || numSegmentsToUpdate != (size_t) segments.size()
                || (size_t) segments[0].getNumSamples() != fftSize * 2)
            {
                segments.clear();

                for (size_t i = 0; i < numSegmentsToUpdate; ++i)
                    segments.push_back ({ 1, static_cast<int> (fftSize * 2) }); // NOLINT
            }
        };

        updateSegmentsIfNecessary (numInputSegments, buffersInputSegments);
        updateSegmentsIfNecessary (numSegments, buffersImpulseSegments);

        auto FFTTempObject = std::make_unique<juce::dsp::FFT> (juce::roundToInt (std::log2 (fftSize)));
        size_t currentPtr = 0;

        for (auto& buf : buffersImpulseSegments)
        {
            buf.clear();

            auto* impulseResponse = buf.getWritePointer (0);

            if (&buf == &buffersImpulseSegments.front())
                impulseResponse[0] = 1.0f;

            juce::FloatVectorOperations::copy (impulseResponse,
                                               samples + currentPtr,
                                               static_cast<int> (juce::jmin (fftSize - blockSize, numSamples - currentPtr)));

            FFTTempObject->performRealOnlyForwardTransform (impulseResponse);
            prepareForConvolution (impulseResponse);

            currentPtr += (fftSize - blockSize);
        }

        reset();
    }

    void reset()
    {
        bufferInput.clear();
        bufferOverlap.clear();
        bufferTempOutput.clear();
        bufferOutput.clear();

        for (auto& buf : buffersInputSegments)
            buf.clear();

        currentSegment = 0;
        inputDataPos = 0;
    }

    void processSamples (const float* input, float* output, size_t numSamples)
    {
        // Overlap-add, zero latency convolution algorithm with uniform partitioning
        size_t numSamplesProcessed = 0;

        auto indexStep = numInputSegments / numSegments;

        auto* inputData = bufferInput.getWritePointer (0);
        auto* outputTempData = bufferTempOutput.getWritePointer (0);
        auto* outputData = bufferOutput.getWritePointer (0);
        auto* overlapData = bufferOverlap.getWritePointer (0);

        while (numSamplesProcessed < numSamples)
        {
            const bool inputDataWasEmpty = (inputDataPos == 0);
            auto numSamplesToProcess = juce::jmin (numSamples - numSamplesProcessed, blockSize - inputDataPos);

            juce::FloatVectorOperations::copy (inputData + inputDataPos, input + numSamplesProcessed, static_cast<int> (numSamplesToProcess));

            auto* inputSegmentData = buffersInputSegments[currentSegment].getWritePointer (0);
            juce::FloatVectorOperations::copy (inputSegmentData, inputData, static_cast<int> (fftSize));

            fftObject->performRealOnlyForwardTransform (inputSegmentData);
            prepareForConvolution (inputSegmentData);

            // Complex multiplication
            if (inputDataWasEmpty)
            {
                juce::FloatVectorOperations::fill (outputTempData, 0, static_cast<int> (fftSize + 1));

                auto index = currentSegment;

                for (size_t i = 1; i < numSegments; ++i)
                {
                    index += indexStep;

                    if (index >= numInputSegments)
                        index -= numInputSegments;

                    convolutionProcessingAndAccumulate (buffersInputSegments[index].getWritePointer (0),
                                                        buffersImpulseSegments[i].getWritePointer (0),
                                                        outputTempData);
                }
            }

            juce::FloatVectorOperations::copy (outputData, outputTempData, static_cast<int> (fftSize + 1));

            convolutionProcessingAndAccumulate (inputSegmentData,
                                                buffersImpulseSegments.front().getWritePointer (0),
                                                outputData);

            updateSymmetricFrequencyDomainData (outputData);
            fftObject->performRealOnlyInverseTransform (outputData);

            // Add overlap
            juce::FloatVectorOperations::add (&output[numSamplesProcessed], &outputData[inputDataPos], &overlapData[inputDataPos], (int) numSamplesToProcess);

            // Input buffer full => Next block
            inputDataPos += numSamplesToProcess;

            if (inputDataPos == blockSize)
            {
                // Input buffer is empty again now
                juce::FloatVectorOperations::fill (inputData, 0.0f, static_cast<int> (fftSize));

                inputDataPos = 0;

                // Extra step for segSize > blockSize
                juce::FloatVectorOperations::add (&(outputData[blockSize]), &(overlapData[blockSize]), static_cast<int> (fftSize - 2 * blockSize));

                // Save the overlap
                juce::FloatVectorOperations::copy (overlapData, &(outputData[blockSize]), static_cast<int> (fftSize - blockSize));

                currentSegment = (currentSegment > 0) ? (currentSegment - 1) : (numInputSegments - 1);
            }

            numSamplesProcessed += numSamplesToProcess;
        }
    }

    void processSamplesWithAddedLatency (const float* input, float* output, size_t numSamples)
    {
        // Overlap-add, zero latency convolution algorithm with uniform partitioning
        size_t numSamplesProcessed = 0;

        auto indexStep = numInputSegments / numSegments;

        auto* inputData = bufferInput.getWritePointer (0);
        auto* outputTempData = bufferTempOutput.getWritePointer (0);
        auto* outputData = bufferOutput.getWritePointer (0);
        auto* overlapData = bufferOverlap.getWritePointer (0);

        while (numSamplesProcessed < numSamples)
        {
            auto numSamplesToProcess = juce::jmin (numSamples - numSamplesProcessed, blockSize - inputDataPos);

            juce::FloatVectorOperations::copy (inputData + inputDataPos, input + numSamplesProcessed, static_cast<int> (numSamplesToProcess));

            juce::FloatVectorOperations::copy (output + numSamplesProcessed, outputData + inputDataPos, static_cast<int> (numSamplesToProcess));

            numSamplesProcessed += numSamplesToProcess;
            inputDataPos += numSamplesToProcess;

            // processing itself when needed (with latency)
            if (inputDataPos == blockSize)
            {
                // Copy input data in input segment
                auto* inputSegmentData = buffersInputSegments[currentSegment].getWritePointer (0);
                juce::FloatVectorOperations::copy (inputSegmentData, inputData, static_cast<int> (fftSize));

                fftObject->performRealOnlyForwardTransform (inputSegmentData);
                prepareForConvolution (inputSegmentData);

                // Complex multiplication
                juce::FloatVectorOperations::fill (outputTempData, 0, static_cast<int> (fftSize + 1));

                auto index = currentSegment;

                for (size_t i = 1; i < numSegments; ++i)
                {
                    index += indexStep;

                    if (index >= numInputSegments)
                        index -= numInputSegments;

                    convolutionProcessingAndAccumulate (buffersInputSegments[index].getWritePointer (0),
                                                        buffersImpulseSegments[i].getWritePointer (0),
                                                        outputTempData);
                }

                juce::FloatVectorOperations::copy (outputData, outputTempData, static_cast<int> (fftSize + 1));

                convolutionProcessingAndAccumulate (inputSegmentData,
                                                    buffersImpulseSegments.front().getWritePointer (0),
                                                    outputData);

                updateSymmetricFrequencyDomainData (outputData);
                fftObject->performRealOnlyInverseTransform (outputData);

                // Add overlap
                juce::FloatVectorOperations::add (outputData, overlapData, static_cast<int> (blockSize));

                // Input buffer is empty again now
                juce::FloatVectorOperations::fill (inputData, 0.0f, static_cast<int> (fftSize));

                // Extra step for segSize > blockSize
                juce::FloatVectorOperations::add (&(outputData[blockSize]), &(overlapData[blockSize]), static_cast<int> (fftSize - 2 * blockSize));

                // Save the overlap
                juce::FloatVectorOperations::copy (overlapData, &(outputData[blockSize]), static_cast<int> (fftSize - blockSize));

                currentSegment = (currentSegment > 0) ? (currentSegment - 1) : (numInputSegments - 1);

                inputDataPos = 0;
            }
        }
    }

    // After each FFT, this function is called to allow convolution to be performed with only 4 SIMD functions calls.
    void prepareForConvolution (float* samples) noexcept
    {
        auto FFTSizeDiv2 = fftSize / 2;

        for (size_t i = 0; i < FFTSizeDiv2; i++)
            samples[i] = samples[i << 1];

        samples[FFTSizeDiv2] = 0;

        for (size_t i = 1; i < FFTSizeDiv2; i++)
            samples[i + FFTSizeDiv2] = -samples[((fftSize - i) << 1) + 1];
    }

    // Does the convolution operation itself only on half of the frequency domain samples.
    void convolutionProcessingAndAccumulate (const float* input, const float* impulse, float* output)
    {
        auto FFTSizeDiv2 = fftSize / 2;
        jassert (juce::isPowerOfTwo (FFTSizeDiv2) && FFTSizeDiv2 > 8);
        jassert (juce::snapPointerToAlignment (input, (size_t) 32) == input);
        jassert (juce::snapPointerToAlignment (impulse, (size_t) 32) == impulse);
        jassert (juce::snapPointerToAlignment (output, (size_t) 32) == output);

        {
            juce::FloatVectorOperations::addWithMultiply (output, input, impulse, static_cast<int> (FFTSizeDiv2));
            juce::FloatVectorOperations::subtractWithMultiply (output, &(input[FFTSizeDiv2]), &(impulse[FFTSizeDiv2]), static_cast<int> (FFTSizeDiv2));

            juce::FloatVectorOperations::addWithMultiply (&(output[FFTSizeDiv2]), input, &(impulse[FFTSizeDiv2]), static_cast<int> (FFTSizeDiv2));
            juce::FloatVectorOperations::addWithMultiply (&(output[FFTSizeDiv2]), &(input[FFTSizeDiv2]), impulse, static_cast<int> (FFTSizeDiv2));
        }

        output[fftSize] += input[fftSize] * impulse[fftSize];
    }

    // Undoes the re-organization of samples from the function prepareForConvolution.
    // Then takes the conjugate of the frequency domain first half of samples to fill the
    // second half, so that the inverse transform will return real samples in the time domain.
    void updateSymmetricFrequencyDomainData (float* samples) noexcept
    {
        auto FFTSizeDiv2 = fftSize / 2;

        for (size_t i = 1; i < FFTSizeDiv2; i++)
        {
            samples[(fftSize - i) << 1] = samples[i];
            samples[((fftSize - i) << 1) + 1] = -samples[FFTSizeDiv2 + i];
        }

        samples[1] = 0.f;

        for (size_t i = 1; i < FFTSizeDiv2; i++)
        {
            samples[i << 1] = samples[(fftSize - i) << 1];
            samples[(i << 1) + 1] = -samples[((fftSize - i) << 1) + 1];
        }
    }

    //==============================================================================
    const size_t blockSize;
    const size_t fftSize;
    const std::unique_ptr<juce::dsp::FFT> fftObject;
    const size_t numSegments;
    const size_t numInputSegments;
    size_t currentSegment = 0, inputDataPos = 0;

    Convolution_Internal_Buffer bufferInput, bufferOutput, bufferTempOutput, bufferOverlap;
    std::vector<Convolution_Internal_Buffer> buffersInputSegments, buffersImpulseSegments;
};

std::vector<float> generate (size_t N, std::mt19937& rng)
{
    std::vector<float> data {};
    data.resize (N);

    std::uniform_real_distribution<float> dist { -1.0f, 1.0f };
    for (auto& x : data)
        x = dist (rng);

    return data;
}

static bool test_convolution (int ir_length_samples, int block_size, int num_blocks, bool latency)
{
    std::cout << "Running test with IR length: " << ir_length_samples
              << ", block size: " << block_size
              << ", latency: " << (latency ? "ON" : "OFF") << '\n';

    std::mt19937 rng { 0x12345 };
    auto ir = generate (ir_length_samples, rng);
    const auto input = generate (block_size * num_blocks, rng);
    std::vector<float> ref_output (input.size());

    ConvolutionEngine reference_engine { ir.data(), ir.size(), (size_t) block_size };
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_blocks; ++i)
    {
        const auto* block_in = input.data() + (i * block_size);
        auto* block_out_ref = ref_output.data() + (i * block_size);
        if (latency)
            reference_engine.processSamplesWithAddedLatency (block_in, block_out_ref, block_size);
        else
            reference_engine.processSamples (block_in, block_out_ref, block_size);
    }
    auto duration = std::chrono::high_resolution_clock::now() - start;
    auto ref_duration_seconds = std::chrono::duration<float> (duration).count();
    std::cout << "  juce::dsp::Convolution: " << ref_duration_seconds << " seconds" << std::endl;

    std::vector<float> test_output (input.size());
    chowdsp::convolution::Config conv_config {};
    chowdsp::convolution::create_config (&conv_config, block_size);
    auto* fft_scratch = (float*) chowdsp::fft::aligned_malloc (conv_config.fft_size * sizeof (float));

    chowdsp::convolution::IR_Uniform conv_ir {};
    chowdsp::convolution::create_ir (&conv_config,
                                     &conv_ir,
                                     ir.data(),
                                     (int) ir.size(),
                                     fft_scratch);

    chowdsp::convolution::Process_Uniform_State conv_state {};
    chowdsp::convolution::create_process_state (&conv_config, &conv_ir, &conv_state);

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_blocks; ++i)
    {
        const auto* block_in = input.data() + (i * block_size);
        auto* block_out_test = test_output.data() + (i * block_size);
        if (latency)
        {
            chowdsp::convolution::process_samples_with_latency (
                &conv_config,
                &conv_ir,
                &conv_state,
                block_in,
                block_out_test,
                block_size,
                fft_scratch);
        }
        else
        {
            chowdsp::convolution::process_samples (&conv_config,
                                                   &conv_ir,
                                                   &conv_state,
                                                   block_in,
                                                   block_out_test,
                                                   block_size,
                                                   fft_scratch);
        }
    }
    duration = std::chrono::high_resolution_clock::now() - start;
    auto test_duration_seconds = std::chrono::duration<float> (duration).count();
    std::cout << "  chowdsp_convolution: " << test_duration_seconds << " seconds" << std::endl;
    std::cout << "  chowdsp is " << ref_duration_seconds / test_duration_seconds << "x faster\n";

    chowdsp::fft::aligned_free (fft_scratch);
    chowdsp::convolution::destroy_ir (&conv_ir);
    chowdsp::convolution::destroy_process_state (&conv_state);
    chowdsp::convolution::destroy_config (&conv_config);

    float error_accum {};
    float max_error {};
    for (int i = 0; i < test_output.size(); ++i)
    {
        const auto ref = ref_output[i];
        const auto test = test_output[i];
        const auto err = ref - test;
        max_error = std::max (max_error, std::abs (err));
        error_accum += err * err;
    }
    const auto mse = error_accum / static_cast<float> (test_output.size());
    std::cout << "  Max error: " << max_error << '\n';
    std::cout << "  Mean-squared error: " << mse << '\n';

    return max_error < 5.0e-4f && mse < 1.0e-9f;
}

int main()
{
    auto success = true;
    for (bool latency : { false, true })
    {
        success &= test_convolution (6000, 2048, 4, latency);
        success &= test_convolution (6000, 512, 20, latency);
        success &= test_convolution (6000, 511, 20, latency);
        success &= test_convolution (6000, 32, 400, latency);
        success &= test_convolution (100, 2048, 2, latency);
        success &= test_convolution (100, 512, 4, latency);
        success &= test_convolution (100, 511, 4, latency);
        success &= test_convolution (100, 32, 10, latency);
    }

#if BUILD_RELEASE
    std::cout << "Speed comparisons:\n";
    success &= test_convolution (48'000, 512, 10'000, false);
    success &= test_convolution (48'000, 512, 10'000, true);
#endif

    return success ? 0 : 1;
}
