#include "chowdsp_convolution.h"

#include <algorithm>
#include <cstring>
#include <chowdsp_fft.h>

namespace chowdsp::convolution
{
static int next_pow2 (int v) noexcept
{
    --v;
    v |= (v >> 1);
    v |= (v >> 2);
    v |= (v >> 4);
    v |= (v >> 8);
    v |= (v >> 16);
    return v + 1;
}

void create_config (Config* config, int max_block_size)
{
    config->block_size = next_pow2 (max_block_size);
    config->fft_size = config->block_size > 128 ? 2 * config->block_size : 4 * config->block_size;
    config->fft = fft::fft_new_setup (config->fft_size, fft::FFT_REAL);
}

void destroy_config (Config* config)
{
    fft::fft_destroy_setup (config->fft);
    *config = {};
}

void create_state (const Config* config, State* state, const float* ir, int ir_num_samples)
{
    size_t state_bytes_needed {};

    const auto segment_num_samples = config->fft_size;
    state->num_segments = (ir_num_samples / (config->fft_size - config->block_size)) + 1;
    state_bytes_needed += (segment_num_samples + state->num_segments) * sizeof (float);

    state->input_num_segments = config->block_size > 128 ? state->num_segments : 3 * state->num_segments;
    state_bytes_needed += (segment_num_samples + state->input_num_segments) * sizeof (float);

    state_bytes_needed += config->fft_size * sizeof (float); // fft scratch
    state_bytes_needed += config->fft_size * sizeof (float); // input data
    state_bytes_needed += config->fft_size * sizeof (float); // output data
    state_bytes_needed += config->fft_size * sizeof (float); // output temp data
    state_bytes_needed += config->fft_size * sizeof (float); // overlap data
    state->data = fft::aligned_malloc (state_bytes_needed);
    float* data = reinterpret_cast<float*> (state->data);

    auto* fft_scratch = data;
    data += config->fft_size;

    state->impulse_segments = data;
    data += segment_num_samples * state->num_segments;
    int current_ptr {};
    for (int seg_idx = 0; seg_idx < state->num_segments; ++seg_idx)
    {
        float* segment = state->impulse_segments + segment_num_samples * seg_idx;
        memset (segment, 0, segment_num_samples * sizeof (float));

        memcpy (segment,
                ir + current_ptr,
                std::min (config->fft_size - config->block_size, ir_num_samples - current_ptr));
        fft::fft_transform_unordered (config->fft,
                                      segment,
                                      segment,
                                      fft_scratch,
                                      fft::FFT_FORWARD);

        current_ptr += config->fft_size - config->block_size;
    }

    state->input_segments = data;
    data += segment_num_samples * state->input_num_segments;
    state->input_data = data;
    data += config->fft_size;
    state->output_data = data;
    data += config->fft_size;
    state->output_temp_data = data;
    data += config->fft_size;
    state->overlap_data = data;
    data += config->fft_size;

    reset (config, state);
}

void destroy_state (State* state)
{
    fft::aligned_free (state->data);
    *state = {};
}

void reset (const Config* config, State* state)
{
    state->current_segment = 0;
    state->input_data_pos = 0;

    const auto segment_num_samples = config->fft_size;
    memset (state->input_segments,
            0,
            segment_num_samples * state->input_num_segments * sizeof (float));
}

void process_samples (const Config* config,
                      State* state,
                      const float* input,
                      float* output,
                      int num_samples)
{
    const auto segment_num_samples = config->fft_size;
    int num_samples_processed = 0;
    auto* fft_scratch = reinterpret_cast<float*> (state->data);
    auto index_step = state->input_num_segments / state->num_segments;

    while (num_samples_processed < num_samples)
    {
        const auto input_data_was_empty = state->input_data_pos == 0;
        const auto samples_to_process = std::min (num_samples - num_samples_processed,
                                                  config->block_size - state->input_data_pos);

        memcpy (state->input_data + state->input_data_pos,
                input + num_samples_processed,
                samples_to_process * sizeof (float));

        auto* input_segment_data = state->input_segments + segment_num_samples * state->current_segment;
        memcpy (input_segment_data, state->input_data, config->fft_size * sizeof (float));

        fft::fft_transform_unordered (config->fft,
                                      input_segment_data,
                                      input_segment_data,
                                      fft_scratch,
                                      fft::FFT_FORWARD);

        // Complex multiplication
        if (input_data_was_empty)
        {
            memset (state->output_temp_data, 0, config->fft_size * sizeof (float));

            auto index = state->current_segment;
            for (int seg_idx = 0; seg_idx < state->num_segments; ++seg_idx)
            {
                index += index_step;
                if (index >= state->input_num_segments)
                    index -= state->input_num_segments;

                const auto* input_segment = state->input_segments + segment_num_samples * index;
                const auto* ir_segment = state->impulse_segments + segment_num_samples * seg_idx;
                fft::fft_convolve_unordered (config->fft,
                                             input_segment,
                                             ir_segment,
                                             state->output_temp_data,
                                             1.0f);
            }
        }

        memcpy (state->output_data, state->output_temp_data, config->fft_size * sizeof (float));

        fft::fft_convolve_unordered (config->fft,
                                     input_segment_data,
                                     state->impulse_segments,
                                     state->output_data,
                                     1.0f);
        fft::fft_transform_unordered (config->fft,
                                      state->output_data,
                                      state->output_data,
                                      fft_scratch,
                                      fft::FFT_BACKWARD);
        const auto fft_inv_scale = 1.0f / static_cast<float> (config->fft_size);
        for (int i = 0; i < config->fft_size; ++i)
            state->output_data[i] *= fft_inv_scale;

        // Add overlap (TODO: SIMD)
        for (int i = 0; i < samples_to_process; ++i)
            output[num_samples_processed + i] = state->output_data[state->input_data_pos + i] + state->overlap_data[state->input_data_pos + i];

        // Input buffer full => Next block
        state->input_data_pos += samples_to_process;

        if (state->input_data_pos == config->block_size)
        {
            // Input buffer is empty again now
            memset (state->input_data, 0, config->fft_size * sizeof (float));

            state->input_data_pos = 0;

            // Extra step for segSize > blockSize
            for (int i = 0; i < config->fft_size - 2 * config->block_size; ++i)
                state->output_data[config->block_size + i] += state->overlap_data[config->block_size + i];

            // Save the overlap
            memcpy (state->overlap_data,
                    state->output_data + config->block_size,
                    (config->fft_size - config->block_size) * sizeof (float));

            state->current_segment = (state->current_segment > 0) ? (state->current_segment - 1) : (state->input_num_segments - 1);
        }

        num_samples_processed += samples_to_process;
    }
}
} // namespace chowdsp::convolution
