#pragma once

namespace chowdsp::convolution
{
struct Config
{
    int block_size {};
    int fft_size {};
    void* fft {};
};

/* State for a single IR (mono). */
struct IR_State
{
    int num_segments {};
    float* segments {};
};

struct Process_State
{
    int num_segments {};
    float* segments {};
    float* input_data {};
    float* output_data {};
    float* output_temp_data {};
    float* overlap_data {};
    int current_segment {};
    int input_data_pos {};
};

/** Creates a convolution config for a given maximum block size */
void create_config (Config*, int max_block_size);
void destroy_config (Config*);

/**
 * Creates a state object for a monophonic IR.
 *
 * The fft_scratch pointer should be point to
 * an array of config->fft_size floats, and should
 * have 64-byte alignment.
 */
void create_ir_state (const Config*, IR_State*, const float* ir, int ir_num_samples, float* fft_scratch);
void create_ir_state (const Config*, IR_State*, int ir_num_samples);
void load_ir_state (const Config*, IR_State*, const float* ir, int ir_num_samples, float* fft_scratch);
void destroy_ir_state (IR_State*);

/** Creates a mono process state object for a given IR. */
void create_process_state (const Config*, const IR_State*, Process_State*);
void reset_process_state (const Config*, Process_State*);
void destroy_process_state (Process_State*);

/**
 * Performs convolution processing for a given IR and state.
 *
 * The fft_scratch pointer should be point to
 * an array of config->fft_size floats, and should
 * have 64-byte alignment.
 */
void process_samples (const Config*,
                      const IR_State*,
                      Process_State*,
                      const float* in,
                      float* out,
                      int N,
                      float* fft_scratch);
} // namespace chowdsp::convolution
