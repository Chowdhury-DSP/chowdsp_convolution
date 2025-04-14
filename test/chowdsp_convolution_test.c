#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <chowdsp_convolution.h>
#include <chowdsp_fft.h>

int main()
{
    printf("Running C tests\n");

    // setup config
    const int block_size = 512;
    struct Convolution_Config conv_config;
    create_config (&conv_config, block_size);
    float* fft_scratch = (float*) aligned_malloc (conv_config.fft_size * sizeof (float));

    // load IR (ideal impulse with delay)
    const int ir_size = 200;
    const int delay_samples = 100;
    float* ir = (float*) calloc(ir_size, sizeof (float));
    ir[delay_samples] = 1.0f;
    struct IR_Uniform conv_ir;
    create_ir (&conv_config, &conv_ir, ir, ir_size, fft_scratch);
    free (ir);

    // set up process state
    struct Process_Uniform_State conv_state;
    create_process_state(&conv_config, &conv_ir, &conv_state);

    // set up i/o buffers
    const int num_blocks = 1000;
    const int data_size = block_size * num_blocks;
    float* test_input_data = malloc (data_size * sizeof (float));
    float* test_output_data = malloc (data_size * sizeof (float));
    for (int i = 0; i < data_size; ++i)
        test_input_data[i] = sinf(314.0f * (float) i / (float) data_size);

    // process convolution
    for (int i = 0; i < num_blocks; ++i)
    {
        const float* block_in = test_input_data + (i * block_size);
        float* block_out = test_output_data + (i * block_size);
        process_samples (&conv_config,
                         &conv_ir,
                         &conv_state,
                         block_in,
                         block_out,
                         block_size,
                         fft_scratch);
    }

    // compute error
    float error_accum = 0.0f;
    float max_error = 0.0f;
    for (int i = 0; i < data_size; ++i)
    {
        const float ref = i < delay_samples ? 0.0f : test_input_data[i - delay_samples];
        const float test = test_output_data[i];
        const float err = fabsf (ref - test);
        
        if (err > max_error)
            max_error = err;
        error_accum += err * err;
    }
    const float mse = error_accum / (float) data_size;
    printf("Max Error: %f\n", max_error);
    printf("Mean-squared: %f\n", mse);

    // cleanup
    free (test_input_data);
    free (test_output_data);
    aligned_free (fft_scratch);
    destroy_process_state (&conv_state);
    destroy_ir (&conv_ir);
    destroy_config (&conv_config);

    return 0;
}
