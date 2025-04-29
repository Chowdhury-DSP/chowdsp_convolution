cmake_minimum_required(VERSION 3.20)
project(chowdsp_convolution VERSION 0.1.0)

if(CHOWDSP_CONVOLUTION_ASAN)
    message(STATUS "Setting flags for address sanitizer: -fsanitize=address -g")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -g")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -g")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")
endif()

if(TARGET chowdsp_fft)
    get_target_property(chowdsp_fft_dir chowdsp_fft SOURCE_DIR)
    message(STATUS "Using chowdsp_fft from ${chowdsp_fft_dir}")
else()
    message(STATUS "Using chowdsp_fft from CPM")
    include(cmake/CPM.cmake)
    unset(JUCE_MODULES_DIR CACHE) # this causes problems with the tests CMake config?
    CPMAddPackage("gh:Chowdhury-DSP/chowdsp_fft#main")
endif()

add_library(chowdsp_convolution STATIC)
target_sources(chowdsp_convolution
    PRIVATE
        chowdsp_convolution.h
        chowdsp_convolution.cpp
)
target_include_directories(chowdsp_convolution PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(chowdsp_convolution PUBLIC chowdsp_fft)
target_compile_features(chowdsp_convolution PRIVATE cxx_std_20)

if(CHOWDSP_CONVOLUTION_TESTING)
    add_subdirectory(test)
endif()
