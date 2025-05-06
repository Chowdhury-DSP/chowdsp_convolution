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

static int pad_floats (int N)
{
    static constexpr int pad_len = 16;
    const auto N_div =  (N + pad_len - 1) / pad_len;
    return N_div * pad_len;
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

//================================================================================================================
void create_ir (const Config* config, IR_Uniform* ir, const float* ir_data, int ir_num_samples, float* fft_scratch)
{
    create_zero_ir (config, ir, ir_num_samples);
    load_ir (config, ir, ir_data, ir_num_samples, fft_scratch);
}

static int get_num_segments (const Config* config, int ir_num_samples)
{
    return (ir_num_samples / (config->fft_size - config->block_size)) + 1;
}

static float* get_segment (const Config* config, float* segments, int segment_idx)
{
    return segments + config->fft_size * segment_idx;
}

static void create_zero_ir_num_segments (const Config* config, IR_Uniform* ir, int num_segments)
{
    size_t bytes_needed {};

    const auto segment_num_samples = config->fft_size;
    ir->max_num_segments = num_segments;
    ir->num_segments = num_segments;
    bytes_needed += segment_num_samples * ir->num_segments * sizeof (float);

    ir->segments = static_cast<float*> (fft::aligned_malloc (bytes_needed));
    memset (ir->segments, 0, ir->num_segments * segment_num_samples * sizeof (float));
}

void create_zero_ir (const Config* config, IR_Uniform* ir, int ir_num_samples)
{
    create_zero_ir_num_segments (config, ir, get_num_segments (config, ir_num_samples));
    ir->num_channels = 1;
}

void load_ir (const Config* config, IR_Uniform* ir, const float* ir_data, int ir_num_samples, float* fft_scratch)
{
    assert (ir->num_channels == 1);

    const auto num_segments = get_num_segments (config, ir_num_samples);
    assert (num_segments <= ir->max_num_segments); // IR is too large for the allocated number of segments
    ir->num_segments = num_segments;

    int current_ptr {};
    for (int seg_idx = 0; seg_idx < ir->num_segments; ++seg_idx)
    {
        float* segment = get_segment (config, ir->segments, seg_idx);
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

void create_multichannel_ir (const Config* config, IR_Uniform* ir, const float* const* ir_data, int ir_num_samples, int num_channels, float* fft_scratch)
{
    create_zero_multichannel_ir (config, ir, ir_num_samples, num_channels);
    load_multichannel_ir (config, ir, ir_data, ir_num_samples, num_channels, fft_scratch);
}

void create_zero_multichannel_ir (const Config* config, IR_Uniform* ir, int ir_num_samples, int num_channels)
{
    const auto mono_ir_num_segments = get_num_segments (config, ir_num_samples);

    create_zero_ir_num_segments (config, ir, mono_ir_num_segments * num_channels);
    assert (ir->num_segments % num_channels == 0);
    const auto actual_num_segments = ir->num_segments / num_channels;
    ir->num_segments = actual_num_segments;
    ir->max_num_segments = actual_num_segments;
    ir->num_channels = num_channels;
}

void load_multichannel_ir (const Config* config, IR_Uniform* ir, const float* const* ir_data, int ir_num_samples, int num_channels, float* fft_scratch)
{
    assert (num_channels == ir->num_channels);

    for (int ch = 0; ch < num_channels; ++ch)
    {
        IR_Uniform this_channel_ir
        {
            .segments = get_segment (config, ir->segments, ch * ir->max_num_segments),
            .num_segments = ir->num_segments,
            .max_num_segments = ir->max_num_segments,
            .num_channels = 1,
        };
        load_ir (config, &this_channel_ir, ir_data[ch], ir_num_samples, fft_scratch);
    }
}

//================================================================================================================
static size_t state_data_bytes_needed (const Config* config, const IR_Uniform* ir, Process_Uniform_State* state)
{
    size_t bytes_needed {};

    const auto segment_num_samples = config->fft_size;
    state->num_channels = ir->num_channels;
    state->max_num_segments = config->block_size > 128 ? ir->max_num_segments : 3 * ir->max_num_segments;
    bytes_needed += segment_num_samples * state->max_num_segments * sizeof (float);

    bytes_needed += config->fft_size * sizeof (float); // input data
    bytes_needed += config->fft_size * sizeof (float); // output data
    bytes_needed += config->fft_size * sizeof (float); // output temp data
    bytes_needed += config->fft_size * sizeof (float); // overlap data
    return bytes_needed * state->num_channels;
}

static void state_data_partition_memory (const Config* config, Process_Uniform_State* state, Process_Uniform_State::State_Data& state_data, float*& data)
{
    const auto segment_num_samples = config->fft_size;

    state_data.segments = data;
    data += segment_num_samples * state->max_num_segments;
    state_data.input_data = data;
    data += config->fft_size;
    state_data.output_data = data;
    data += config->fft_size;
    state_data.output_temp_data = data;
    data += config->fft_size;
    state_data.overlap_data = data;
    data += config->fft_size;
}

void create_process_state (const Config* config, const IR_Uniform* ir, Process_Uniform_State* state)
{
    using State_Data = Process_Uniform_State::State_Data;
    const auto state_bytes_needed = state_data_bytes_needed (config, ir, state);
    auto* data = fft::aligned_malloc (state_bytes_needed + state->num_channels * sizeof (State_Data));
    state->state_data = reinterpret_cast<State_Data*> (static_cast<std::byte*> (data) + state_bytes_needed);

    auto* float_data = static_cast<float*> (data);
    for (int ch = 0; ch < state->num_channels; ++ch)
        state_data_partition_memory (config, state, state->state_data[ch], float_data);
    assert (static_cast<void*> (float_data) == static_cast<void*> (state->state_data));

    reset_process_state (config, state);
}

void reset_process_state (const Config* config, Process_Uniform_State* state)
{
    state->current_segment = 0;
    state->input_data_pos = 0;

    const auto segment_num_samples = config->fft_size;
    for (int ch = 0; ch < state->num_channels; ++ch)
    {
        auto& state_data = state->state_data[ch];
        memset (state_data.segments,
                0,
                segment_num_samples * state->max_num_segments * sizeof (float));

        memset (state_data.input_data, 0, config->fft_size * sizeof (float));
        memset (state_data.output_data, 0, config->fft_size * sizeof (float));
        memset (state_data.output_temp_data, 0, config->fft_size * sizeof (float));
        memset (state_data.overlap_data, 0, config->fft_size * sizeof (float));
    }
}

void destroy_process_state (Process_Uniform_State* state)
{
    fft::aligned_free (state->state_data[0].segments);
    *state = {};
}

//================================================================================================================
int get_required_nuir_scratch_bytes (const IR_Non_Uniform* ir)
{
    assert (ir->head_config != nullptr);
    assert (ir->tail_config != nullptr);
    return static_cast<int> ((std::max (ir->head_config->fft_size,
                     ir->tail_config->fft_size)
           + pad_floats (ir->head_config->block_size)) * sizeof (float));
}

void create_nuir (IR_Non_Uniform* ir, const float* ir_data, int ir_num_samples, float* fft_scratch)
{
    create_zero_nuir (ir, ir_num_samples);
    load_nuir (ir, ir_data, ir_num_samples, fft_scratch);
}

void create_zero_nuir (IR_Non_Uniform* ir, int ir_num_samples)
{
    assert (ir->head_config != nullptr);
    assert (ir->tail_config != nullptr);
    assert (ir->head_size >= ir->head_config->block_size);
    assert (ir->tail_config->block_size == ir->head_size);
    assert (ir_num_samples >= 2 * ir->head_size);

    const auto head_num_segments = get_num_segments (ir->head_config, ir->head_size);
    const auto head_segments_length = head_num_segments * ir->head_config->fft_size;
    const auto tail_num_segments = get_num_segments (ir->tail_config, ir_num_samples - ir->head_size);
    const auto tail_segments_length = tail_num_segments * ir->tail_config->fft_size;
    const auto total_segments_length = head_segments_length + tail_segments_length;

    auto* segment_data = static_cast<float*> (fft::aligned_malloc (total_segments_length * sizeof (float)));
    memset (segment_data, 0, total_segments_length * sizeof (float));

    ir->head.segments = segment_data;
    ir->head.num_segments = head_num_segments;
    ir->head.max_num_segments = head_num_segments;
    ir->head.num_channels = 1;
    ir->tail.segments = segment_data + head_segments_length;
    ir->tail.num_segments = tail_num_segments;
    ir->tail.max_num_segments = tail_num_segments;
    ir->tail.num_channels = 1;
}

void load_nuir (IR_Non_Uniform* ir, const float* ir_data, int ir_num_samples, float* fft_scratch)
{
    load_ir (ir->head_config, &ir->head, ir_data, std::min (ir_num_samples, ir->head_size), fft_scratch);
    load_ir (ir->tail_config, &ir->tail, ir_data + ir->head_size, std::max (ir_num_samples - ir->head_size, 0), fft_scratch);
}

void destroy_nuir (IR_Non_Uniform* ir)
{
    fft::aligned_free (ir->head.segments);
    *ir = {};
}

//================================================================================================================
void create_nuir_process_state (const IR_Non_Uniform* ir, Process_Non_Uniform_State* state)
{
    using State_Data = Process_Uniform_State::State_Data;

    state->head_config = ir->head_config;
    state->tail_config = ir->tail_config;

    const auto head_state_bytes_needed = state_data_bytes_needed (state->head_config, &ir->head, &state->head);
    const auto tail_state_bytes_needed = state_data_bytes_needed (state->tail_config, &ir->tail, &state->tail);
    auto* data = fft::aligned_malloc (head_state_bytes_needed + tail_state_bytes_needed + 2 * sizeof (State_Data));
    state->head.state_data = reinterpret_cast<State_Data*> (static_cast<std::byte*> (data) + head_state_bytes_needed + tail_state_bytes_needed);
    state->tail.state_data = state->head.state_data + 1;

    auto* float_data = static_cast<float*>(data);

    state_data_partition_memory (state->head_config, &state->head, state->head.state_data[0], float_data);
    state_data_partition_memory (state->tail_config, &state->tail, state->tail.state_data[0], float_data);
    assert (static_cast<void*> (float_data) == static_cast<void*> (state->head.state_data));

    reset_process_state (state->head_config, &state->head);
    reset_process_state (state->tail_config, &state->tail);
}

void reset_nuir_process_state (Process_Non_Uniform_State* state)
{
    reset_process_state (state->head_config, &state->head);
    reset_process_state (state->tail_config, &state->tail);
}

void destroy_nuir_process_state (Process_Non_Uniform_State* state)
{
    destroy_process_state (&state->head);
    *state = {};
}

//================================================================================================================
static void process_samples_mono (const Config* config,
                                  const IR_Uniform* ir,
                                  Process_Uniform_State* state,
                                  const float* input,
                                  float* output,
                                  int num_samples,
                                  float* fft_scratch)
{
    const auto fft_inv_scale = 1.0f / static_cast<float> (config->fft_size);
    const auto state_num_segments = config->block_size > 128 ? ir->num_segments : 3 * ir->num_segments;
    auto index_step = state_num_segments / ir->num_segments;
    state->current_segment = (state->current_segment >= state_num_segments) ? 0 : state->current_segment;
    auto* state_data = state->state_data;

    int num_samples_processed = 0;
    while (num_samples_processed < num_samples)
    {
        const auto input_data_was_empty = state->input_data_pos == 0;
        const auto samples_to_process = std::min (num_samples - num_samples_processed,
                                                  config->block_size - state->input_data_pos);

        memcpy (state_data->input_data + state->input_data_pos,
                input + num_samples_processed,
                samples_to_process * sizeof (float));

        auto* input_segment_data = get_segment (config, state_data->segments, state->current_segment);
        memcpy (input_segment_data, state_data->input_data, config->fft_size * sizeof (float));

        fft::fft_transform_unordered (config->fft,
                                      input_segment_data,
                                      input_segment_data,
                                      fft_scratch,
                                      fft::FFT_FORWARD);

        // Complex multiplication
        if (input_data_was_empty)
        {
            memset (state_data->output_temp_data, 0, config->fft_size * sizeof (float));

            auto index = state->current_segment;
            for (int seg_idx = 1; seg_idx < ir->num_segments; ++seg_idx)
            {
                index += index_step;
                if (index >= state_num_segments)
                    index -= state_num_segments;

                const auto* input_segment = get_segment (config, state_data->segments, index);
                const auto* ir_segment = get_segment (config, ir->segments, seg_idx);
                fft::fft_convolve_unordered (config->fft,
                                             input_segment,
                                             ir_segment,
                                             state_data->output_temp_data,
                                             fft_inv_scale);
            }
        }

        memcpy (state_data->output_data, state_data->output_temp_data, config->fft_size * sizeof (float));

        fft::fft_convolve_unordered (config->fft,
                                     input_segment_data,
                                     ir->segments,
                                     state_data->output_data,
                                     fft_inv_scale);
        fft::fft_transform_unordered (config->fft,
                                      state_data->output_data,
                                      state_data->output_data,
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
                output[num_samples_processed + i] = state_data->output_data[state->input_data_pos + i] + state_data->overlap_data[state->input_data_pos + i];
        }

        // Input buffer full => Next block
        state->input_data_pos += samples_to_process;

        if (state->input_data_pos == config->block_size)
        {
            // Input buffer is empty again now
            memset (state_data->input_data, 0, config->fft_size * sizeof (float));

            state->input_data_pos = 0;

            // Extra step for segSize > blockSize
            const auto extra_block_samples = config->fft_size - 2 * config->block_size;
            if (extra_block_samples > 0)
            {
                fft::fft_accumulate (config->fft,
                                     state_data->overlap_data + config->block_size,
                                     state_data->output_data + config->block_size,
                                     state_data->output_data + config->block_size,
                                     extra_block_samples);
            }

            // Save the overlap
            memcpy (state_data->overlap_data,
                    state_data->output_data + config->block_size,
                    (config->fft_size - config->block_size) * sizeof (float));

            state->current_segment = (state->current_segment > 0) ? (state->current_segment - 1) : (state_num_segments - 1);
        }

        num_samples_processed += samples_to_process;
    }
}

void process_samples (const Config* config,
                      const IR_Uniform* ir,
                      Process_Uniform_State* state,
                      const float* input,
                      float* output,
                      int num_samples,
                      float* fft_scratch)
{
    assert (ir->num_channels == 1);
    process_samples_mono (config, ir, state, input, output, num_samples, fft_scratch);
}

static void process_samples_with_latency_mono (const Config* config,
                                               const IR_Uniform* ir,
                                               Process_Uniform_State* state,
                                               const float* input,
                                               float* output,
                                               int num_samples,
                                               float* fft_scratch)
{
    const auto fft_inv_scale = 1.0f / static_cast<float> (config->fft_size);
    const auto state_num_segments = config->block_size > 128 ? ir->num_segments : 3 * ir->num_segments;
    auto index_step = state_num_segments / ir->num_segments;
    state->current_segment = (state->current_segment >= state_num_segments) ? 0 : state->current_segment;
    auto* state_data = state->state_data;

    int num_samples_processed = 0;
    while (num_samples_processed < num_samples)
    {
        const auto samples_to_process = std::min (num_samples - num_samples_processed,
                                                  config->block_size - state->input_data_pos);

        memcpy (state_data->input_data + state->input_data_pos,
                input + num_samples_processed,
                samples_to_process * sizeof (float));

        memcpy (output + num_samples_processed,
                state_data->output_data + state->input_data_pos,
                samples_to_process * sizeof (float));

        num_samples_processed += samples_to_process;
        state->input_data_pos += samples_to_process;

        if (state->input_data_pos == config->block_size)
        {
            // Copy input data in input segment
            auto* input_segment_data = get_segment (config, state_data->segments, state->current_segment);
            memcpy (input_segment_data, state_data->input_data, config->fft_size * sizeof (float));

            fft::fft_transform_unordered (config->fft,
                                          input_segment_data,
                                          input_segment_data,
                                          fft_scratch,
                                          fft::FFT_FORWARD);

            // Complex multiplication
            memset (state_data->output_temp_data, 0, config->fft_size * sizeof (float));

            auto index = state->current_segment;
            for (int seg_idx = 1; seg_idx < ir->num_segments; ++seg_idx)
            {
                index += index_step;
                if (index >= state_num_segments)
                    index -= state_num_segments;

                const auto* input_segment = get_segment (config, state_data->segments, index);
                const auto* ir_segment = get_segment (config, ir->segments, seg_idx);
                fft::fft_convolve_unordered (config->fft,
                                             input_segment,
                                             ir_segment,
                                             state_data->output_temp_data,
                                             fft_inv_scale);
            }

            memcpy (state_data->output_data, state_data->output_temp_data, config->fft_size * sizeof (float));

            fft::fft_convolve_unordered (config->fft,
                                         input_segment_data,
                                         ir->segments,
                                         state_data->output_data,
                                         fft_inv_scale);
            fft::fft_transform_unordered (config->fft,
                                          state_data->output_data,
                                          state_data->output_data,
                                          fft_scratch,
                                          fft::FFT_BACKWARD);

            // Add overlap
            fft::fft_accumulate (config->fft,
                                 state_data->overlap_data,
                                 state_data->output_data,
                                 state_data->output_data,
                                 config->block_size);

            // Input buffer is empty again now
            memset (state_data->input_data, 0, config->fft_size * sizeof (float));

            // Extra step for segSize > blockSize
            const auto extra_block_samples = config->fft_size - 2 * config->block_size;
            if (extra_block_samples > 0)
            {
                fft::fft_accumulate (config->fft,
                                     state_data->overlap_data + config->block_size,
                                     state_data->output_data + config->block_size,
                                     state_data->output_data + config->block_size,
                                     extra_block_samples);
            }

            // Save the overlap
            memcpy (state_data->overlap_data,
                    state_data->output_data + config->block_size,
                    (config->fft_size - config->block_size) * sizeof (float));

            state->current_segment = (state->current_segment > 0) ? (state->current_segment - 1) : (state_num_segments - 1);

            state->input_data_pos = 0;
        }
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
    assert (ir->num_channels == 1);
    process_samples_with_latency_mono (config, ir, state, input, output, num_samples, fft_scratch);
}

static void process_multichannel (const Config* config,
                                  const IR_Uniform* ir,
                                  Process_Uniform_State* state,
                                  const float* const* in,
                                  float* const* out,
                                  int N,
                                  int num_channels,
                                  float* fft_scratch,
                                  bool with_latency)
{
    assert (state->num_channels == ir->num_channels);
    assert (state->num_channels == num_channels);

    for (int ch = 0; ch < num_channels; ++ch)
    {
        IR_Uniform mono_ir
        {
            .segments = get_segment (config, ir->segments, ch * ir->max_num_segments),
            .num_segments = ir->num_segments,
            .max_num_segments = ir->max_num_segments,
            .num_channels = 1,
        };

        Process_Uniform_State mono_state
        {
            .state_data = state->state_data + ch,
            .max_num_segments = state->max_num_segments,
            .current_segment = state->current_segment,
            .input_data_pos = state->input_data_pos,
            .num_channels = 1,
        };

        if (with_latency)
            process_samples_with_latency_mono (config, &mono_ir, &mono_state, in[ch], out[ch], N, fft_scratch);
        else
            process_samples_mono (config, &mono_ir, &mono_state, in[ch], out[ch], N, fft_scratch);

        if (ch == num_channels - 1)
        {
            state->current_segment = mono_state.current_segment;
            state->input_data_pos = mono_state.input_data_pos;
        }
    }
}

void process_samples_multichannel (const Config* config,
                                   const IR_Uniform* ir,
                                   Process_Uniform_State* state,
                                   const float* const* in,
                                   float* const* out,
                                   int N,
                                   int num_channels,
                                   float* fft_scratch)
{
    process_multichannel (config, ir, state, in, out, N, num_channels, fft_scratch, false);
}

void process_samples_with_latency_multichannel (const Config* config,
                                                const IR_Uniform* ir,
                                                Process_Uniform_State* state,
                                                const float* const* in,
                                                float* const* out,
                                                int N,
                                                int num_channels,
                                                float* fft_scratch)
{
    process_multichannel (config, ir, state, in, out, N, num_channels, fft_scratch, true);
}

void process_samples_non_uniform (const IR_Non_Uniform* ir,
                                  Process_Non_Uniform_State* state,
                                  const float* in,
                                  float* out,
                                  int N,
                                  float* scratch)
{
    auto* tail_out = scratch;
    scratch += pad_floats (N);

    process_samples_with_latency (ir->tail_config,
                                  &ir->tail,
                                  &state->tail,
                                  in,
                                  tail_out,
                                  N,
                                  scratch);

    process_samples (ir->head_config,
                     &ir->head,
                     &state->head,
                     in,
                     out,
                     N,
                     scratch);

    for (int n = 0; n < N; ++n)
        out[n] += tail_out[n];
}
} // namespace chowdsp::convolution
