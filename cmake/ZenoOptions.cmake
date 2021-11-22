# https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
set(ZENO_TARGET "Editor" CACHE STRING "Specify the Zeno target desired to build (Editor, Headless, Benchmark, Tests)")
option(ZENO_WITH_SYCL "Enable SYCL support for Zeno" OFF)
option(ZENO_WITH_LEGACY "Build Zeno With Legacy Nodes" OFF)
option(ZENO_WITH_BACKWARD "Enable stack backtrace for Zeno" OFF)

############### BEGIN ADHOC ###############
if (UNIX)  # these are only used by archibate and zhxx1987

    if ($ENV{HOME} STREQUAL "/home/bate")
        message(STATUS "BATE detected, making him happy")
        include(cmake/BATE.cmake)
    elseif ($ENV{HOME} STREQUAL "/home/dilei")
        message(STATUS "ZHXX detected, making him happy")
        include(cmake/ZHXX.cmake)
    endif()

endif()  # normal users won't be affected
################ END ADHOC ################

if (UNIX)
    find_program(CCACHE_PROGRAM ccache)
    if (CCACHE_PROGRAM)
        message(STATUS "Found CCache: ${CCACHE_PROGRAM}")
        set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
        set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "${CCACHE_PROGRAM}")
    endif()
endif()

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_POSITION_INDEPENDENT_CODE OFF)
#set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(BUILD_SHARED_LIBS OFF)
if (WIN32)
    add_definitions(-DNOMINMAX -D_USE_MATH_DEFINES)
endif()
