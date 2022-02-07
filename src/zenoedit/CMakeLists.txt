find_package(Qt5 COMPONENTS Widgets OpenGL REQUIRED)
message(STATUS "Found Qt5 version ${Qt5_VERSION}: ${Qt5_DIR}")

file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp *.ui *.qrc)
add_executable(zenoedit ${source}) 

set_property(TARGET zenoedit PROPERTY AUTOUIC ON)
set_property(TARGET zenoedit PROPERTY AUTOMOC ON)
set_property(TARGET zenoedit PROPERTY AUTORCC ON)
set_property(TARGET zenoedit PROPERTY WIN32_EXECUTABLE ON)

if (ZENO_FAULTHANDLER)
    target_sources(zenoedit PRIVATE ${BACKWARD_ENABLE})
    add_backward(zenoedit)
endif()

target_link_libraries(zenoedit PRIVATE Qt5::Widgets Qt5::OpenGL)
target_link_libraries(zenoedit PRIVATE zenoui zenoio zenovis glad zeno zeno_external_include ${CMAKE_DL_LIBS})
target_include_directories(zenoedit PRIVATE . .. ../zenoui)
