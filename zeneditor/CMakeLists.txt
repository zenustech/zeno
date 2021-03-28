file(GLOB PROJECT_SOURCE *.cpp *.h)

add_library(zeneditor SHARED)
target_include_directories(zeneditor PRIVATE ../external)
target_link_directories(zeneditor PRIVATE ../external)

target_sources(zeneditor PRIVATE ${PROJECT_SOURCE})
target_include_directories(zeneditor PRIVATE .)
target_link_libraries(zeneditor PRIVATE OpenGL GLEW glfw imgui imnodes)

include(../scripts/Pybind11.cmake)
do_python_bindings()
