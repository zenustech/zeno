add_library(zeno_EasyGL OBJECT)
target_include_directories(zeno_EasyGL PRIVATE .)

target_link_libraries(zeno_EasyGL PUBLIC zenocore)
target_link_libraries(zeno_EasyGL PUBLIC GLESv2)
target_sources(zeno_EasyGL PRIVATE nodes.cpp passtoy.cpp stb_image.cpp)

option(EASYGL_ENABLE_GLUT "Enable GLUT nodes in ZENO EasyGL" ON)
if (EASYGL_ENABLE_GLUT)
    target_sources(zeno_EasyGL PRIVATE glut.cpp)
    target_link_libraries(zeno_EasyGL PUBLIC glut)
endif()

#add_executable(zeno_EasyGL_main main.cpp)
#target_link_libraries(zeno_EasyGL_main PRIVATE zenocore)
#target_link_libraries(zeno_EasyGL_main PRIVATE zeno_EasyGL)
#target_link_libraries(zeno_EasyGL_main PRIVATE GLESv2)
#target_link_libraries(zeno_EasyGL_main PRIVATE glut)
