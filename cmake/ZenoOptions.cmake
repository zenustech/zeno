# https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
set(HIPSYCL_TARGETS "omp;cuda:sm_52;cuda:sm_61;cuda:sm_70;cuda:sm_75;cuda:sm_86" CACHE STRING "Specify the hipSYCL targets to build against")
set(ZENO_TARGET "Editor" CACHE STRING "Specify the Zeno target desired to build (Editor, Client, Benchmark, Tests)")
option(ZENO_WITH_SYCL "Enable SYCL support for Zeno" OFF)
option(ZENO_WITH_LEGACY "Build Zeno With Legacy Nodes" OFF)
option(ZENO_WITH_BACKWARD "Enable stack backtrace for Zeno" OFF)

############### BEGIN ADHOC ###############
if (UNIX)  # these are only used by archibate and zhxx1987

    if ($ENV{HOME} STREQUAL "/home/dilei")
        message("-- ZHXX detected, making him happy")
        set(HIPSYCL_TARGETS "omp;cuda:sm_86")
        set(ZENO_WITH_SYCL OFF)
        set(ZENO_WITH_LEGACY ON)
        set(ZENO_WITH_BACKWARD ON)
        set(ZENO_TARGET Editor)
        set(CMAKE_BUILD_TYPE Release)

    elseif ($ENV{HOME} STREQUAL "/home/bate")
        message("-- BATE detected, making him happy")
        set(HIPSYCL_TARGETS "omp")
        set(ZENO_WITH_SYCL OFF)
        set(ZENO_WITH_LEGACY OFF)
        set(ZENO_WITH_BACKWARD ON)
        set(ZENO_TARGET Benchmark)
        set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} /usr/local/lib/cmake/hipSYCL)
        set(CMAKE_BUILD_TYPE Debug)
    endif()

endif()  # normal users won't be affected
################ END ADHOC ################

if (UNIX)
    find_program(CCACHE_PROGRAM ccache)
    if (CCACHE_PROGRAM)
        message("-- Found CCache: ${CCACHE_PROGRAM}")
        set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ${CCACHE_PROGRAM})
        set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ${CCACHE_PROGRAM})
    endif()
endif()

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
message("-- Build type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(BUILD_SHARED_LIBS OFF)
if (WIN32)
    add_definitions(-DNOMINMAX -D_USE_MATH_DEFINES)
endif()
