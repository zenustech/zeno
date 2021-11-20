# https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
set(HIPSYCL_TARGETS "omp;cuda:sm_52;cuda:sm_61;cuda:sm_70;cuda:sm_75;cuda:sm_86" CACHE STRING "Specify the hipSYCL targets to build against")
set(ZENO_TARGET "Editor" CACHE STRING "Specify the Zeno target desired to build (Editor, Headless, Benchmark, Tests)")
option(ZENO_WITH_SYCL "Enable SYCL support for Zeno" OFF)
option(ZENO_WITH_LEGACY "Build Zeno With Legacy Nodes" OFF)
option(ZENO_WITH_BACKWARD "Enable stack backtrace for Zeno" OFF)
option(ZENO_WITH_ZPM "Use ZPM to manage Zeno dependencies" ON)

############### BEGIN ADHOC ###############
if (UNIX)  # these are only used by archibate and zhxx1987

    if ($ENV{HOME} STREQUAL "/home/bate")
        message("-- BATE detected, making him happy")
        include(cmake/BATE.cmake)
    elseif ($ENV{HOME} STREQUAL "/home/dilei")
        message("-- ZHXX detected, making him happy")
        include(cmake/ZHXX.cmake)
    endif()

endif()  # normal users won't be affected
################ END ADHOC ################

if (UNIX)
    find_program(CCACHE_PROGRAM ccache)
    if (CCACHE_PROGRAM)
        message("-- Found CCache: ${CCACHE_PROGRAM}")
        set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
        set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "${CCACHE_PROGRAM}")
    endif()
endif()

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
message("-- Build type: ${CMAKE_BUILD_TYPE}")

if (NOT DEFINED ZPM_INSTALL_PREFIX)
    set(ZPM_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/installed")
endif()
message("-- ZPM directory: [${ZPM_INSTALL_PREFIX}]")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_POSITION_INDEPENDENT_CODE OFF)
#set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(BUILD_SHARED_LIBS OFF)
if (WIN32)
    add_definitions(-DNOMINMAX -D_USE_MATH_DEFINES)
endif()
