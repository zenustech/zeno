add_library(zeno.core OBJECT)
target_include_directories(zeno.core PUBLIC .)
target_link_libraries(zeno.core PUBLIC spdlog)
target_link_libraries(zeno.core PRIVATE glad glfw ftgl glm)

file(GLOB_RECURSE source *.h *.cpp)
target_sources(zeno.core PRIVATE ${source})
