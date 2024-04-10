file(GLOB_RECURSE source CONFIGURE_DEPENDS src/*.cpp)
add_library(zenocore STATIC ${source})
target_include_directories(zenocore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include ${CMAKE_CURRENT_SOURCE_DIR}/../tpls/include)
target_link_libraries(zeno PUBLIC zenocore)