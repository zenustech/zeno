file(GLOB_RECURSE sources CONFIGURE_DEPENDS *.h *.cpp)

add_executable(zenorunner ${sources})
target_link_libraries(zenorunner PUBLIC zeno zenoinclude)
