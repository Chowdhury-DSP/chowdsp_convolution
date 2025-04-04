include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/CPM.cmake)

CPMAddPackage("gh:juce-framework/juce#8.0.6")
CPMAddPackage("gh:Chowdhury-DSP/chowdsp_utils#next")
CPMAddPackage("gh:Chowdhury-DSP/chowdsp_fft#main")

if(NOT TARGET chowdsp_fft_juce)
    juce_add_module(${chowdsp_fft_SOURCE_DIR}/chowdsp_fft_juce)
endif()

add_executable(chowdsp_convolution_test chowdsp_convolution_test.cpp)
target_link_libraries(chowdsp_convolution_test
    # juce::juce_dsp
    juce::juce_audio_basics
    juce::juce_audio_formats
    chowdsp::chowdsp_buffers
    chowdsp_fft_juce
    chowdsp_convolution
)
target_compile_definitions(chowdsp_convolution_test
    PRIVATE
        JUCE_MODULE_AVAILABLE_juce_dsp=1 # chowdsp_pffft includes juce_dsp internally!
)
target_compile_features(chowdsp_convolution_test PRIVATE cxx_std_20)
