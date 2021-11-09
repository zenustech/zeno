file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp)

add_executable(zeno ${source})
