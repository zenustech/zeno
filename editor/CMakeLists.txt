find_package(Qt5 COMPONENTS Widgets OpenGL REQUIRED)
message(STATUS "Found Qt5 version ${Qt5_VERSION}: ${Qt5_DIR}")

file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp *.ui *.qrc)
add_executable(zeno_editor ${source}) 

set_property(TARGET zeno_editor PROPERTY AUTOUIC ON)
set_property(TARGET zeno_editor PROPERTY AUTOMOC ON)
set_property(TARGET zeno_editor PROPERTY AUTORCC ON)
set_property(TARGET zeno_editor PROPERTY WIN32_EXECUTABLE ON)

if (ZENO_WITH_BACKWARD)
    target_sources(zeno_editor PRIVATE ${BACKWARD_ENABLE})
    add_backward(zeno_editor)
endif()

target_link_libraries(zeno_editor PRIVATE Qt5::Widgets Qt5::OpenGL)
target_link_libraries(zeno_editor PRIVATE zenoui zenoio zenvis glad zeno)

target_include_directories(zeno_editor PRIVATE . ../zenoui .. ../zeno/zeno)
