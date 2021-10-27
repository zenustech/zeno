add_library(zeno OBJECT)
target_include_directories(zeno PUBLIC .)
target_include_directories(zeno PUBLIC zeno/legacy)

if (BATE)
    target_compile_options(zeno PUBLIC -Wl,-rpath=/opt/sycl/lib -std=c++17 -fsycl -fsycl-targets=nvptx64-nvidia-cuda)
    target_compile_definitions(zeno PUBLIC -DZENO_WITH_SYCL)
endif()

target_link_libraries(zeno PUBLIC spdlog)

file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp)
target_sources(zeno PRIVATE ${source})
