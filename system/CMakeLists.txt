file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp *.ui *.qrc)
add_executable(zeno_system ${source})

target_link_libraries(zeno_system PRIVATE zeno)