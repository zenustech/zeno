target_include_directories(zeno PUBLIC .)

file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp)
target_sources(zeno PRIVATE ${source})

if (SYCL_TARGETS)
    message("-- Using Clang for SYCL with target(s): [${SYCL_TARGETS}]")
    target_compile_definitions(zeno PUBLIC -DZENO_WITH_SYCL)
else()
    message(WARNING "-- No SYCL backend specified, using host emulated SYCL")
endif()
