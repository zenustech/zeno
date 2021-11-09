file(GLOB_RECURSE source RELATIVE client CONFIGURE_DEPENDS *.h *.cpp)

add_executable(zeno ${source})
