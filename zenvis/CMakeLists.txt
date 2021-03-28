file(GLOB PROJECT_SOURCE "*.cpp" "*.hpp" "*.h")

add_executable(zenvis)
target_include_directories(zenvis PRIVATE ../external)
target_link_directories(zenvis PRIVATE ../external)

target_sources(zenvis PRIVATE ${PROJECT_SOURCE})
target_include_directories(zenvis PRIVATE .)
target_link_libraries(zenvis PRIVATE OpenGL GLEW glfw imgui)
