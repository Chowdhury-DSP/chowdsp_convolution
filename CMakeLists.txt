cmake_minimum_required(VERSION 3.20)
project(chowdsp_convolution)

if(NOT TARGET chowdsp_fft)
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

add_subdirectory(test)
