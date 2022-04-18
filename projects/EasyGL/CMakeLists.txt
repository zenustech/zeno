target_include_directories(zeno PRIVATE .)

target_link_libraries(zeno PUBLIC zeno)
target_link_libraries(zeno PUBLIC GLESv2)
target_sources(zeno PRIVATE nodes.cpp passtoy.cpp stb_image.cpp)

option(EASYGL_ENABLE_GLUT "Enable GLUT nodes in ZENO EasyGL" ON)
if (EASYGL_ENABLE_GLUT)
    target_sources(zeno PRIVATE glut.cpp)
    target_link_libraries(zeno PUBLIC glut)
endif()
