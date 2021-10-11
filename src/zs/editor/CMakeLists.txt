target_link_libraries(zs PUBLIC glad glfw ftgl glm spdlog)
target_link_libraries(zs PUBLIC tinyfiledialogs rapidjson)

file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp)
target_sources(zs PRIVATE ${source})
