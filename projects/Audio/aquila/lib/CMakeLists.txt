# Ooura FFT
enable_language(C)
set(Ooura_fft_SOURCES ooura/fft4g.c)

add_library(Ooura_fft ${Ooura_fft_SOURCES})

# UnitTest++
if(Aquila_BUILD_TESTS)
    add_subdirectory(unittestpp)
endif()
