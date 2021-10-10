add_library(zs.editor STATIC)
target_link_libraries(zs.editor PUBLIC glad glfw ftgl glm spdlog)
target_link_libraries(zs.editor PUBLIC tinyfiledialogs rapidjson)
target_link_libraries(zs.editor PUBLIC zs.ztd zs.assets)

file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp)
target_sources(zs.editor PRIVATE ${source})
