find_package(Qt5 COMPONENTS Widgets OpenGL Network Quick Svg REQUIRED)
find_package(Threads REQUIRED)

message(STATUS "Found Qt5 version ${Qt5_VERSION}: ${Qt5_DIR}")

#add_custom_target(
#    clean_resource_file
#    COMMAND ${CMAKE_COMMAND} -E remove -f ${PROJECT_BINARY_DIR}/ui/zenoedit/qrc_font.o
#    COMMAND ${CMAKE_COMMAND} -E remove -f ${PROJECT_BINARY_DIR}/ui/zenoedit/qrc_icons.o
#)

file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp *.ui)
file(GLOB_RECURSE resource CONFIGURE_DEPENDS *.qrc)
file(GLOB_RECURSE bigresource CONFIGURE_DEPENDS *font.qrc)

FOREACH(item ${resource})
    IF(item IN_LIST bigresource)
        LIST(REMOVE_ITEM resource ${item})
    ENDIF(item IN_LIST bigresource)
ENDFOREACH(item)

qt5_add_big_resources(source ${bigresource})  # cihou msvc: fatal error C1060: compiler is out of heap space
qt5_add_resources(source ${resource})

add_executable(zenqt ${source})
# Add a dependency to ensure that the resource file is deleted before being regenerated

# add_dependencies(zenqt clean_resource_file)

set_property(TARGET zenqt PROPERTY AUTOUIC ON)
set_property(TARGET zenqt PROPERTY AUTOMOC ON)
set_property(TARGET zenqt PROPERTY AUTORCC ON)
set_property(TARGET zenqt PROPERTY WIN32_EXECUTABLE OFF)

if (ZENO_DEBUG_MSVC)
    zeno_dbg_msvc(zenqt)
endif()


target_link_libraries(zenqt PRIVATE Threads::Threads)
target_link_libraries(zenqt PRIVATE Qt5::Widgets Qt5::OpenGL Qt5::Svg Qt5::Network Qt5::Quick qt5advanceddocking QCodeEditor)

if (WIN32)
    target_link_libraries(zenqt PRIVATE Dbghelp Version)
    target_link_libraries(zenqt PRIVATE ws2_32.lib winmm.lib wldap32.lib Advapi32.lib Crypt32.lib)
endif()

target_link_libraries(zenqt PRIVATE zenocore zenovis)
target_include_directories(zenqt PRIVATE . ./include ../3rdparty/include ../3rdparty/Qt-Advanced-Docking-System/src)

add_custom_command(TARGET zenqt PRE_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${PROJECT_SOURCE_DIR}/hdr ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hdr)