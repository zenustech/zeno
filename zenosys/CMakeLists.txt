file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp *.ui *.qrc)
add_executable(zenosys ${source})

target_link_libraries(zenosys PRIVATE zeno)
