file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp *.ui)

add_executable(zeno ${source})
