add_library(zs.zeno STATIC)
target_include_directories(zs.zeno PUBLIC .)
target_link_libraries(zs.zeno PUBLIC glad glfw ftgl glm spdlog)
target_link_libraries(zs.zeno PUBLIC tinyfiledialogs rapidjson)
target_link_libraries(zs.zeno PUBLIC zs.zeno.ztd zs.zeno.assets)

file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp)
target_sources(zs.zeno PRIVATE ${source})
