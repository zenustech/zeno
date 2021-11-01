target_include_directories(zeno PUBLIC .)

file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp)
target_sources(zeno PRIVATE ${source})

if (ZENO_WITH_SYCL)
    message("-- Building Zeno with SYCL")
    target_compile_definitions(zeno PUBLIC -DZENO_WITH_SYCL)
else()
    message(WARNING "Building Zeno with host emulated SYCL")
endif()
