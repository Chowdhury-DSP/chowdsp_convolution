#pragma once

#ifdef __cplusplus
extern "C"
{
namespace chowdsp::convolution
{
#endif

/**
 * Convolution configuration.
 * This depends only on the maximum block size.
 */
struct Convolution_Config
{
    int block_size;
    int fft_size;
    void* fft;
};
#ifdef __cplusplus
using Config = Convolution_Config;
#endif

/** State for a mono uniform-partitioned IR. */
struct IR_Uniform
{
    float* segments;
    int num_segments;
    int max_num_segments;
};

/** State for processing a mono uniform-partitioned IR */
struct Process_Uniform_State
{
    float* segments;
    float* input_data;
    float* output_data;
    float* output_temp_data;
    float* overlap_data;
    int max_num_segments;
    int current_segment;
    int input_data_pos;
};

/** Creates a convolution config for a given maximum block size */
void create_config (struct Convolution_Config*, int max_block_size);

/** De-allocates the config's internal data. */
void destroy_config (struct Convolution_Config*);

/**
 * Creates a monophonic IR.
 *
 * The fft_scratch pointer should be point to
 * an array of config->fft_size floats, and should
 * have 64-byte alignment.
 */
void create_ir (const struct Convolution_Config*, struct IR_Uniform*, const float* ir, int ir_num_samples, float* fft_scratch);

/**
 * Creates a mono IR of a given size.
 * The IR will be filled with zeros.
 */
void create_zero_ir (const struct Convolution_Config*, struct IR_Uniform*, int ir_num_samples);

/**
 * Loads IR data.
 * The data must be the same number of samples as the IR was created with.
 */
void load_ir (const struct Convolution_Config*, struct IR_Uniform*, const float* ir, int ir_num_samples, float* fft_scratch);

/** De-allocates the IR's internal data. */
void destroy_ir (struct IR_Uniform*);

/** Creates a mono process state object for a given IR. */
void create_process_state (const struct Convolution_Config*, const struct IR_Uniform*, struct Process_Uniform_State*);

/** Zeros the process state. */
void reset_process_state (const struct Convolution_Config*, struct Process_Uniform_State*);

/** De-allocates the state's internal data. */
void destroy_process_state (struct Process_Uniform_State*);

/**
 * Performs convolution processing for a given IR and state.
 *
 * The fft_scratch pointer should be point to
 * an array of config->fft_size floats, and should
 * have 64-byte alignment.
 */
void process_samples (const struct Convolution_Config*,
                      const struct IR_Uniform*,
                      struct Process_Uniform_State*,
                      const float* in,
                      float* out,
                      int N,
                      float* fft_scratch);

/**
 * Similar to process_samples(), but with an added
 * config->block_size samples of latency. In exchange,
 * the convolution processing will be a little bit
 * faster, especially when processing with odd block
 * sizes.
 */
void process_samples_with_latency (const struct Convolution_Config*,
                                   const struct IR_Uniform*,
                                   struct Process_Uniform_State*,
                                   const float* in,
                                   float* out,
                                   int N,
                                   float* fft_scratch);

#ifdef __cplusplus
} // namespace chowdsp::convolution
} // extern "C"
#endif
