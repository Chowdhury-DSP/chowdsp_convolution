#pragma once

namespace chowdsp::convolution
{
struct Config
{
    int block_size {};
    int fft_size {};
    void* fft {};
    // should we put the FFT scratch here?
};

// This state is MONO (for now)
struct State
{
    int num_segments {};
    int input_num_segments {};
    float* impulse_segments {};
    float* input_segments {};
    float* input_data {};
    float* output_data {};
    float* output_temp_data {};
    float* overlap_data {};
    void* data {};
    int current_segment {};
    int input_data_pos {};
};

void create_config (Config*, int max_block_size);
void destroy_config (Config*);

void create_state (const Config*, State*, const float* ir, int ir_num_samples);
void destroy_state (State*);

void reset (const Config*, State*);
void process_samples (const Config*, State*, const float* in, float* out, int N);
}
