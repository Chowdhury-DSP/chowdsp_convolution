#include "chowdsp_convolution.h"

#include <algorithm>
#include <cassert>
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

void create_ir (const Config* config, IR_Uniform* ir, const float* ir_data, int ir_num_samples, float* fft_scratch)
{
    create_zero_ir (config, ir, ir_num_samples);
    load_ir (config, ir, ir_data, ir_num_samples, fft_scratch);
}

void create_zero_ir (const Config* config, IR_Uniform* ir, int ir_num_samples)
{
    size_t bytes_needed {};

    const auto segment_num_samples = config->fft_size;
    const auto num_segments = (ir_num_samples / (config->fft_size - config->block_size)) + 1;
    ir->max_num_segments = num_segments;
    ir->num_segments = num_segments;
    bytes_needed += segment_num_samples * ir->num_segments * sizeof (float);

    ir->segments = static_cast<float*> (fft::aligned_malloc (bytes_needed));
    memset (ir->segments, 0, ir->num_segments * segment_num_samples * sizeof (float));
}

void load_ir (const Config* config, IR_Uniform* ir, const float* ir_data, int ir_num_samples, float* fft_scratch)
{
    const auto segment_num_samples = config->fft_size;

    const auto num_segments = (ir_num_samples / (config->fft_size - config->block_size)) + 1;
    assert (num_segments <= ir->max_num_segments); // IR is too large for the allocated number of segments
    ir->num_segments = num_segments;

    int current_ptr {};
    for (int seg_idx = 0; seg_idx < ir->num_segments; ++seg_idx)
    {
        float* segment = ir->segments + segment_num_samples * seg_idx;
        memcpy (segment,
                ir_data + current_ptr,
                std::min (config->fft_size - config->block_size, ir_num_samples - current_ptr) * sizeof (float));
        fft::fft_transform_unordered (config->fft,
                                      segment,
                                      segment,
                                      fft_scratch,
                                      fft::FFT_FORWARD);

        current_ptr += config->fft_size - config->block_size;
    }
}

void destroy_ir (IR_Uniform* ir)
{
    fft::aligned_free (ir->segments);
    *ir = {};
}

void create_process_state (const Config* config, const IR_Uniform* ir, Process_Uniform_State* state)
{
    size_t bytes_needed {};

    const auto segment_num_samples = config->fft_size;

    state->max_num_segments = config->block_size > 128 ? ir->max_num_segments : 3 * ir->max_num_segments;
    bytes_needed += segment_num_samples * state->max_num_segments * sizeof (float);

    bytes_needed += config->fft_size * sizeof (float); // input data
    bytes_needed += config->fft_size * sizeof (float); // output data
    bytes_needed += config->fft_size * sizeof (float); // output temp data
    bytes_needed += config->fft_size * sizeof (float); // overlap data
    auto* data = static_cast<float*> (fft::aligned_malloc (bytes_needed));

    state->segments = data;
    data += segment_num_samples * state->max_num_segments;
    state->input_data = data;
    data += config->fft_size;
    state->output_data = data;
    data += config->fft_size;
    state->output_temp_data = data;
    data += config->fft_size;
    state->overlap_data = data;
    data += config->fft_size;

    reset_process_state (config, state);
}

void reset_process_state (const Config* config, Process_Uniform_State* state)
{
    state->current_segment = 0;
    state->input_data_pos = 0;

    const auto segment_num_samples = config->fft_size;
    memset (state->segments,
            0,
            segment_num_samples * state->max_num_segments * sizeof (float));

    memset (state->input_data, 0, config->fft_size * sizeof (float));
    memset (state->output_data, 0, config->fft_size * sizeof (float));
    memset (state->output_temp_data, 0, config->fft_size * sizeof (float));
    memset (state->overlap_data, 0, config->fft_size * sizeof (float));
}

void destroy_process_state (Process_Uniform_State* state)
{
    fft::aligned_free (state->segments);
    *state = {};
}

void process_samples (const Config* config,
                      const IR_Uniform* ir,
                      Process_Uniform_State* state,
                      const float* input,
                      float* output,
                      int num_samples,
                      float* fft_scratch)
{
    const auto fft_inv_scale = 1.0f / static_cast<float> (config->fft_size);
    const auto segment_num_samples = config->fft_size;
    const auto state_num_segments = config->block_size > 128 ? ir->num_segments : 3 * ir->num_segments;
    auto index_step = state_num_segments / ir->num_segments;

    int num_samples_processed = 0;
    while (num_samples_processed < num_samples)
    {
        const auto input_data_was_empty = state->input_data_pos == 0;
        const auto samples_to_process = std::min (num_samples - num_samples_processed,
                                                  config->block_size - state->input_data_pos);

        memcpy (state->input_data + state->input_data_pos,
                input + num_samples_processed,
                samples_to_process * sizeof (float));

        auto* input_segment_data = state->segments + segment_num_samples * state->current_segment;
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
            for (int seg_idx = 1; seg_idx < ir->num_segments; ++seg_idx)
            {
                index += index_step;
                if (index >= state_num_segments)
                    index -= state_num_segments;

                const auto* input_segment = state->segments + segment_num_samples * index;
                const auto* ir_segment = ir->segments + segment_num_samples * seg_idx;
                fft::fft_convolve_unordered (config->fft,
                                             input_segment,
                                             ir_segment,
                                             state->output_temp_data,
                                             fft_inv_scale);
            }
        }

        memcpy (state->output_data, state->output_temp_data, config->fft_size * sizeof (float));

        fft::fft_convolve_unordered (config->fft,
                                     input_segment_data,
                                     ir->segments,
                                     state->output_data,
                                     fft_inv_scale);
        fft::fft_transform_unordered (config->fft,
                                      state->output_data,
                                      state->output_data,
                                      fft_scratch,
                                      fft::FFT_BACKWARD);

        // Add overlap
        {
            // Using SIMD for this operation is tricky, because
            // we can't guarantee that the pointers will be aligned.

            // const auto vec_width_x2 = 2 * fft::fft_simd_width_bytes (config->fft) / static_cast<int> (sizeof (float));
            // const auto n_samples_vec = (samples_to_process / vec_width_x2) * vec_width_x2;
            // fft::fft_accumulate (config->fft,
            //                      state->output_data + state->input_data_pos,
            //                      state->overlap_data + state->input_data_pos,
            //                      output + num_samples_processed,
            //                      n_samples_vec);
            // for (int i = n_samples_vec; i < samples_to_process; ++i) // extra data that can't be SIMD-ed
            //     output[num_samples_processed + i] = state->output_data[state->input_data_pos + i] + state->overlap_data[state->input_data_pos + i];

            for (int i = 0; i < samples_to_process; ++i)
                output[num_samples_processed + i] = state->output_data[state->input_data_pos + i] + state->overlap_data[state->input_data_pos + i];
        }

        // Input buffer full => Next block
        state->input_data_pos += samples_to_process;

        if (state->input_data_pos == config->block_size)
        {
            // Input buffer is empty again now
            memset (state->input_data, 0, config->fft_size * sizeof (float));

            state->input_data_pos = 0;

            // Extra step for segSize > blockSize
            const auto extra_block_samples = config->fft_size - 2 * config->block_size;
            if (extra_block_samples > 0)
            {
                fft::fft_accumulate (config->fft,
                                     state->overlap_data + config->block_size,
                                     state->output_data + config->block_size,
                                     state->output_data + config->block_size,
                                     extra_block_samples);
            }

            // Save the overlap
            memcpy (state->overlap_data,
                    state->output_data + config->block_size,
                    (config->fft_size - config->block_size) * sizeof (float));

            state->current_segment = (state->current_segment > 0) ? (state->current_segment - 1) : (state_num_segments - 1);
        }

        num_samples_processed += samples_to_process;
    }
}

void process_samples_with_latency (const Config* config,
                                   const IR_Uniform* ir,
                                   Process_Uniform_State* state,
                                   const float* input,
                                   float* output,
                                   int num_samples,
                                   float* fft_scratch)
{
    const auto fft_inv_scale = 1.0f / static_cast<float> (config->fft_size);
    const auto segment_num_samples = config->fft_size;
    const auto state_num_segments = config->block_size > 128 ? ir->num_segments : 3 * ir->num_segments;
    auto index_step = state_num_segments / ir->num_segments;

    int num_samples_processed = 0;
    while (num_samples_processed < num_samples)
    {
        const auto samples_to_process = std::min (num_samples - num_samples_processed,
                                                  config->block_size - state->input_data_pos);

        memcpy (state->input_data + state->input_data_pos,
                input + num_samples_processed,
                samples_to_process * sizeof (float));

        memcpy (output + num_samples_processed,
                state->output_data + state->input_data_pos,
                samples_to_process * sizeof (float));

        num_samples_processed += samples_to_process;
        state->input_data_pos += samples_to_process;

        if (state->input_data_pos == config->block_size)
        {
            // Copy input data in input segment
            auto* input_segment_data = state->segments + segment_num_samples * state->current_segment;
            memcpy (input_segment_data, state->input_data, config->fft_size * sizeof (float));

            fft::fft_transform_unordered (config->fft,
                                          input_segment_data,
                                          input_segment_data,
                                          fft_scratch,
                                          fft::FFT_FORWARD);

            // Complex multiplication
            memset (state->output_temp_data, 0, config->fft_size * sizeof (float));

            auto index = state->current_segment;
            for (int seg_idx = 1; seg_idx < ir->num_segments; ++seg_idx)
            {
                index += index_step;
                if (index >= state_num_segments)
                    index -= state_num_segments;

                const auto* input_segment = state->segments + segment_num_samples * index;
                const auto* ir_segment = ir->segments + segment_num_samples * seg_idx;
                fft::fft_convolve_unordered (config->fft,
                                             input_segment,
                                             ir_segment,
                                             state->output_temp_data,
                                             fft_inv_scale);
            }

            memcpy (state->output_data, state->output_temp_data, config->fft_size * sizeof (float));

            fft::fft_convolve_unordered (config->fft,
                                         input_segment_data,
                                         ir->segments,
                                         state->output_data,
                                         fft_inv_scale);
            fft::fft_transform_unordered (config->fft,
                                          state->output_data,
                                          state->output_data,
                                          fft_scratch,
                                          fft::FFT_BACKWARD);

            // Add overlap
            fft::fft_accumulate (config->fft,
                                 state->overlap_data,
                                 state->output_data,
                                 state->output_data,
                                 config->block_size);

            // Input buffer is empty again now
            memset (state->input_data, 0, config->fft_size * sizeof (float));

            // Extra step for segSize > blockSize
            const auto extra_block_samples = config->fft_size - 2 * config->block_size;
            if (extra_block_samples > 0)
            {
                fft::fft_accumulate (config->fft,
                                     state->overlap_data + config->block_size,
                                     state->output_data + config->block_size,
                                     state->output_data + config->block_size,
                                     extra_block_samples);
            }

            // Save the overlap
            memcpy (state->overlap_data,
                    state->output_data + config->block_size,
                    (config->fft_size - config->block_size) * sizeof (float));

            state->current_segment = (state->current_segment > 0) ? (state->current_segment - 1) : (state_num_segments - 1);

            state->input_data_pos = 0;
        }
    }
}
} // namespace chowdsp::convolution
