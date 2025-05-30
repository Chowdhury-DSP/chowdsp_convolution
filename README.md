# chowdsp_convolution

[![Test](https://github.com/Chowdhury-DSP/chowdsp_convolution/actions/workflows/test.yml/badge.svg)](https://github.com/Chowdhury-DSP/chowdsp_convolution/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/Chowdhury-DSP/chowdsp_convolution/graph/badge.svg?token=3WCDKPHA58)](https://codecov.io/gh/Chowdhury-DSP/chowdsp_convolution)

`chowdsp_convolution` is a library for performing frequency-domain
convolution using [`chowdsp_fft`](https://github.com/Chowdhury-DSP/chowdsp_fft).
The library currently supports uniformly-partitioned convolutions,
as well as 2-stage non-uniformly-partitioned convolutions.

**N.B.: This library is still in early development, and
there will likely be breaking changes.** If you have
suggestions for ways to improve the API, or features to
add please create a GitHub Issue.

## Usage

### Basic Usage (mono IR, mono i/o, uniform partitioning)

First, create a `Config` object:

```cpp
chowdsp::convolution::Config config {};
chowdsp::convolution::create_config (&config, maximum_block_size);
```

We'll also allocate some "scratch" data that will be used for computing
FFTs under the hood:

```cpp
float* fft_scratch = chowdsp::fft::aligned_malloc (config->fft_size * sizeof (float));
```

Next, create a partitioned IR:

```cpp
chowdsp::convolution::IR_Uniform ir {};
chowdsp::convolution::create_ir (&config, &ir, my_ir.data(), my_ir.size());
```

Then we'll create a convolution "state".
```cpp
chowdsp::convolution::Process_Uniform_State state {};
chowdsp::convolution::create_process_state (&config, &ir, &state);
```

Now we're ready to process some data:

```cpp
chowdsp::convolution::process_samples (&config, &ir, &state, data, data, num_samples, fft_scratch);
```

Alternatively, we could use `process_samples_with_latency()` which is
faster, but adds `config->block_size` samples of latency.

Finally, let's clean up all our memory allocation:

```cpp
chowdsp::fft::aligned_free (fft_scratch);
chowdsp::convolution::destroy_process_state (&state);
chowdsp::convolution::destroy_ir (&ir);
chowdsp::convolution::destroy_config (&config);
```

### Multi-Channel Processing (mono IR)

Let's say that you want to convolve a stereo audio stream with a mono IR.
We can use `create_multichannel_process_state()` to create a processing state
with a given number of channels.

```cpp
chowdsp::convolution::Process_Uniform_State stereo_state {};
chowdsp::convolution::create_multichannel_process_state (&config, &ir, &stereo_state, 2);
```

To process our audio, we'll want to use `process_samples_multichannel()`
(or `process_samples_multichannel_with_latency()`).

```cpp
float* channel_data[2] {
    left_channel_data,
    right_channel_data,
};
chowdsp::convolution::process_samples_multichannel (&config, &ir, &state, channel_data, channel_data, num_samples, 2, fft_scratch);
```

### Multi-Channel IRs

let's create a stereo, uniform-partitioned IR:

```cpp
float* ir_data[2] {
    left_ir_data,
    right_ir_data,
};
chowdsp::convolution::IR_Uniform ir {};
chowdsp::convolution::create_multichannel_ir (&config, &ir, ir_data, ir_num_samples, 2, fft_scratch);
```

Now if we call `create_process_state()`, the state will automatically be created
for the same number of channels as the IR.
```cpp
chowdsp::convolution::Process_Uniform_State state {};
chowdsp::convolution::create_process_state (&config, &ir, &state);
```

Then (as before), we can do our multi-channel processing:

```cpp
float* channel_data[2] {
    left_channel_data,
    right_channel_data,
};
chowdsp::convolution::process_samples_multichannel (&config, &ir, &state, channel_data, channel_data, num_samples, 2, fft_scratch);
```

### Multi-Threaded Usage

What should you do if you're looking to load an impulse response
on some thread *other* than the audio thread, while the audio
thread is still running? The basic idea is that you should:
- Create a `IR_Uniform` object on your background thread.
- Create one `Process_Uniform_State` object per-channel on your background thread
  - This step may be skipped if the new IR is the same length as the one currently on the audio thread.
- Pass these objects to your audio thread (e.g. via a lock-free queue)
- Pass the old IR and state objects to your background thread where they can be safely destroyed.

Note that the `Config` object is thread-safe, so you may use the
same config on both your audio thread and background thread (e.g.
when calling `create_ir()` or `load_ir()`). However, the `fft_scratch`
is **not** thread-safe, so make sure to allocate a dedicated `fft_scratch`
for each thread.

## License

`chowdsp_convolution` is licensed under the BSD 3-clause license. Enjoy!

### Disclaimer

This implementation is *loosely* based on some code from the
[JUCE](https://github.com/juce-framework/juce) library. Personally,
I think that I've changed enough of the code that this library should
be considered an original work, rather than a "fork" of the JUCE
implementation. That said, if you want to use this library in a
commercial product and you don't have a JUCE license, I'd recommend
looking through both codebases and deciding for yourself.

-- Jatin
