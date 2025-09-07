#pragma once

#ifdef __cplusplus
#include <cstddef>

extern "C"
{
namespace chowdsp::convolution
{
#else
#include <stddef.h>
#include <stdbool.h>
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

/** State for a uniform-partitioned IR. */
struct IR_Uniform
{
    float* segments;
    int num_segments;
    int max_num_segments;
    int num_channels;
};

/** State for processing a uniform-partitioned IR */
struct Process_Uniform_State
{
    struct State_Data
    {
        float* segments;
        float* input_data;
        float* output_data;
        float* output_temp_data;
        float* overlap_data;
    }* state_data;
    int max_num_segments;
    int current_segment;
    int input_data_pos;
    int num_channels;
};

/** State for processing a multi-channel uniform-partitioned IR */
struct Process_Multichannel_Uniform_State
{
    struct Process_Uniform_State state;
    int num_channels;
};

/** State for a mono non-uniform-partitioned IR. */
struct IR_Non_Uniform
{
    struct IR_Uniform head;
    struct IR_Uniform tail;
    const struct Convolution_Config* head_config;
    const struct Convolution_Config* tail_config;
    int head_size;
};

/** State for processing a mono non-uniform-partitioned IR */
struct Process_Non_Uniform_State
{
    struct Process_Uniform_State head;
    struct Process_Uniform_State tail;
    const struct Convolution_Config* head_config;
    const struct Convolution_Config* tail_config;
};

/** Returns the required FFT size for a given block size. */
int convolution_fft_size (int max_block_size);

/** The number of bytes required for `create_config()` with in-place construction. */
size_t config_bytes_required (int max_block_size);

/**
 * Creates a convolution config for a given maximum block size.
 * If no `place_data` pointer is provided, the config will allocate
 * its own memory, and the user must call `destroy_config()` to free
 * that memory. If a `place_data` pointer is provided, the config will
 * be constructed in-place, using the provided memory, and the user is
 * responsible for managing that memory themselves. The `place_data` pointer
 * must provide the number of bytes determined by `config_bytes_required()`,
 * and should be aligned to 64 bytes.
 */
void create_config (struct Convolution_Config*, int max_block_size, void* place_data
#ifdef __cplusplus
     = nullptr
#endif
);

/** De-allocates the config's internal data. */
void destroy_config (struct Convolution_Config*);

/**
 * Returns the number of bytes required to call `create_ir()`
 * or `create_zero_ir()` with `place_data`.
 */
size_t ir_bytes_required (int max_block_size, int ir_num_samples);

/**
 * Creates a monophonic IR.
 *
 * The fft_scratch pointer should point to
 * an array of config->fft_size floats, and should
 * have 64-byte alignment.
 *
 * If `place_data` is provided, the IR will be constructed in-place.
 * Otherwise, memory will be allocated, and the user must call `destroy_ir()`
 * to free that memory. `place_data` should be aligned to 64 bytes.
 */
void create_ir (const struct Convolution_Config*, struct IR_Uniform*, const float* ir, int ir_num_samples, float* fft_scratch, void* place_data
#ifdef __cplusplus
     = nullptr
#endif
);

/**
 * Creates a mono IR of a given size.
 * The IR will be filled with zeros.
 *
 * See the requirements for `place_data` for `create_ir()`.
 */
void create_zero_ir (const struct Convolution_Config*, struct IR_Uniform*, int ir_num_samples, void* place_data
#ifdef __cplusplus
     = nullptr
#endif
);

/**
 * Loads IR data.
 * `ir_num_samples` must be less than or equal the number of samples
 * the IR was created to expect.
 */
void load_ir (const struct Convolution_Config*, struct IR_Uniform*, const float* ir, int ir_num_samples, float* fft_scratch);

/**
 * Returns the number of bytes required to call `create_multichannel_ir()`
 * or `create_zero_multichannel_ir()` with `place_data`.
 */
size_t multichannel_ir_bytes_required (int max_block_size, int ir_num_samples, int num_channels);

/**
 * Creates a multi-channel uniform-partitioned IR.
 *
 * The fft_scratch pointer should point to
 * an array of config->fft_size floats, and should
 * have 64-byte alignment.
 *
 * See the requirements for `place_data` for `create_ir()`.
 */
void create_multichannel_ir (const struct Convolution_Config*, struct IR_Uniform*, const float* const* ir, int ir_num_samples, int num_channels, float* fft_scratch, void* place_data
#ifdef __cplusplus
     = nullptr
#endif
);

/**
 * Creates a multi-channel IR of a given size.
 * The IR will be filled with zeros.
 *
 * See the requirements for `place_data` for `create_ir()`.
 */
void create_zero_multichannel_ir (const struct Convolution_Config*, struct IR_Uniform*, int ir_num_samples, int num_channels, void* place_data
#ifdef __cplusplus
     = nullptr
#endif
);

/**
 * Loads IR data.
 * `ir_num_samples` must be less than or equal the number of samples
 * the IR was created to expect.
 */
void load_multichannel_ir (const struct Convolution_Config*, struct IR_Uniform*, const float* const* ir, int ir_num_samples, int num_channels, float* fft_scratch);

/** De-allocates the IR's internal data. */
void destroy_ir (struct IR_Uniform*);

/**
 * Returns the number of bytes required to call `create_process_state()`
 * with `place_data`.
 */
size_t process_state_bytes_required (int max_block_size, int ir_num_samples);

/**
 * Creates a process state object for a given IR.
 * The process state will be created to process the same number of channels as the IR contains.
 *
 * If `place_data` is provided, the state will be constructed in-place.
 * Otherwise, memory will be allocated, and the user must call `destroy_process_state()`
 * to free that memory. `place_data` should be aligned to 64 bytes.
 */
void create_process_state (const struct Convolution_Config*, const struct IR_Uniform*, struct Process_Uniform_State*, void* place_data
#ifdef __cplusplus
     = nullptr
#endif
);

/**
 * Returns the number of bytes required to call
 * `create_multichannel_process_state()` with `place_data`.
 */
size_t multichannel_process_state_bytes_required (int max_block_size, int ir_num_samples, int num_channels);

/**
 * Creates a process state object for a given IR, with a specific number of channels.
 * This is useful for convolving a monophonic IR with multiple channels.
 *
 * See the requirements for `place_data` for `create_process_state()`.
 */
void create_multichannel_process_state (const struct Convolution_Config*, const struct IR_Uniform*, struct Process_Uniform_State*, int num_channels, void* place_data
#ifdef __cplusplus
     = nullptr
#endif
);

/** Zeros the process state. */
void reset_process_state (const struct Convolution_Config*, struct Process_Uniform_State*);

/** Zeros the process state. */
void reset_process_state_segments (const struct Convolution_Config*, struct Process_Uniform_State*, const struct IR_Uniform*);

/** De-allocates the state's internal data. */
void destroy_process_state (struct Process_Uniform_State*);

/**
 * Creates a monophonic non-uniform IR.
 *
 * The scratch pointer should point to an allocated block
 * of at least get_required_nuir_scratch_bytes(), and should
 * have 64-byte alignment.
 */
void create_nuir (struct IR_Non_Uniform*, const float* ir, int ir_num_samples, float* scratch);

/**
 * Creates a mono non-uniform IR of a given size.
 * The IR will be filled with zeros.
 */
void create_zero_nuir (struct IR_Non_Uniform*, int ir_num_samples);

/** Returns the required scratch size needed for this non-uniform IR. */
int get_required_nuir_scratch_bytes (const struct IR_Non_Uniform*);

/**
 * Loads IR data.
 * `ir_num_samples` must be less than or equal the number of samples
 * the IR was created to expect.
 */
void load_nuir (struct IR_Non_Uniform*, const float* ir, int ir_num_samples, float* scratch);

/** De-allocates the IR's internal data. */
void destroy_nuir (struct IR_Non_Uniform*);

/** Creates a mono process state object for a given IR. */
void create_nuir_process_state (const struct IR_Non_Uniform*, struct Process_Non_Uniform_State*);

/** Zeros the process state. */
void reset_nuir_process_state (struct Process_Non_Uniform_State*);

/** De-allocates the state's internal data. */
void destroy_nuir_process_state (struct Process_Non_Uniform_State*);

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

/**
 * Performs convolution processing for a given multi-channel IR and state.
 *
 * The fft_scratch pointer should be point to
 * an array of config->fft_size floats, and should
 * have 64-byte alignment.
 */
void process_samples_multichannel (const struct Convolution_Config*,
                                   const struct IR_Uniform*,
                                   struct Process_Uniform_State*,
                                   const float* const* in,
                                   float* const* out,
                                   int N,
                                   int num_channels,
                                   float* fft_scratch);

/**
 * Similar to process_samples_multichannel(), but with an added
 * config->block_size samples of latency. In exchange,
 * the convolution processing will be a little bit
 * faster, especially when processing with odd block
 * sizes.
 */
void process_samples_with_latency_multichannel (const struct Convolution_Config*,
                                                const struct IR_Uniform*,
                                                struct Process_Uniform_State*,
                                                const float* const* in,
                                                float* const* out,
                                                int N,
                                                int num_channels,
                                                float* fft_scratch);

/**
 * Performs convolution processing for a given non-uniform IR and state.
 *
 * The scratch pointer should point to an allocated block
 * of at least get_required_nuir_scratch_bytes(), and should
 * have 64-byte alignment.
 */
void process_samples_non_uniform (const struct IR_Non_Uniform*,
                                  struct Process_Non_Uniform_State*,
                                  const float* in,
                                  float* out,
                                  int N,
                                  float* scratch);

#ifdef __cplusplus
} // namespace chowdsp::convolution
} // extern "C"
#endif
