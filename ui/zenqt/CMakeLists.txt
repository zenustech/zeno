find_package(Qt5 COMPONENTS Widgets OpenGL Network Quick QuickWidgets QuickControls2 Svg REQUIRED)
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

set(qmlsource
    qml/testGraphEditor.qml
    qml/Graph.qml
    qml/ZNode.qml
    qml/ZParam.qml
    qml/container/TabView/TabViewController.qml
    qml/container/TabView/TabPages/PagesController.qml
)

FOREACH(item ${resource})
    IF(item IN_LIST bigresource)
        LIST(REMOVE_ITEM resource ${item})
    ENDIF(item IN_LIST bigresource)
ENDFOREACH(item)

qt5_add_big_resources(source ${bigresource})  # cihou msvc: fatal error C1060: compiler is out of heap space
qt5_add_resources(source ${resource})

add_executable(zeno ${source} ${qmlsource})
# Add a dependency to ensure that the resource file is deleted before being regenerated

# add_dependencies(zeno clean_resource_file)

if(MSVC)
    target_compile_options(zeno PRIVATE /MP)
endif()

set_property(TARGET zeno PROPERTY AUTOUIC ON)
set_property(TARGET zeno PROPERTY AUTOMOC ON)
set_property(TARGET zeno PROPERTY AUTORCC ON)
set_property(TARGET zeno PROPERTY WIN32_EXECUTABLE OFF)
set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS $<$<CONFIG:Debug>:QT_QML_DEBUG>)


if (ZENO_DEBUG_MSVC)
    zeno_dbg_msvc(zeno)
endif()

if (MSVC)
    target_compile_options(zeno PRIVATE /MP)
endif()

target_link_libraries(zeno PRIVATE Threads::Threads)
target_link_libraries(zeno PRIVATE Qt5::Widgets Qt5::OpenGL Qt5::Svg Qt5::Network Qt5::Quick Qt5::QuickWidgets qt5advanceddocking QCodeEditor Qt${QT_VERSION_MAJOR}::QuickControls2)

if (WIN32)
    target_link_libraries(zeno PRIVATE Dbghelp Version)
    target_link_libraries(zeno PRIVATE ws2_32.lib winmm.lib wldap32.lib Advapi32.lib Crypt32.lib)
endif()

target_link_libraries(zeno PRIVATE zenobase zenovis)

if (ZENO_WITH_PYTHON)
    find_package(Python3 COMPONENTS Development REQUIRED)
    target_link_libraries(zeno PRIVATE ${Python3_LIBRARIES} zenpybind)
    target_include_directories(zeno PRIVATE ${Python3_INCLUDE_DIRS})
    target_compile_definitions(zeno PRIVATE -DZENO_WITH_PYTHON)
endif()

target_link_libraries(zeno PUBLIC ScintillaEditBase)

if (ZENO_WITH_VLD)
    target_include_directories(zeno PRIVATE "C:/vld2/src")
    target_link_libraries(zeno PRIVATE "C:/vld2/src/bin/x64/Debug-v143/vld.lib")
    target_compile_definitions(zeno PRIVATE -DZENO_WITH_VLD)
endif()

target_include_directories(zeno PUBLIC Qt${QT_VERSION_MAJOR}::QuickControls2 ScintillaEditBase ScintillaEdit)
target_include_directories(zeno PRIVATE 
    .
    ./include
    ../3rdparty/include
    ../3rdparty/Qt-Advanced-Docking-System/src
    ./qml/QuickQanava-2.4/src
    ../3rdparty/QuickScintilla/scintilla/qt/ScintillaEdit
    ../3rdparty/QuickScintilla/scintilla/qt/ScintillaEditBase
    ../3rdparty/QuickScintilla/scintilla/include
    ../3rdparty/QuickScintilla/scintilla/lexilla/src
    ../3rdparty/QuickScintilla/scintilla/lexlib
    ../3rdparty/QuickScintilla/scintilla/src
)

add_custom_command(TARGET zeno PRE_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${PROJECT_SOURCE_DIR}/hdr ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hdr)

if (ZENO_WITH_PYTHON)
    target_compile_definitions(zeno PRIVATE -DZENO_WITH_PYTHON)
endif()