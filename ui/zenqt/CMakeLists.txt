find_package(Qt5 COMPONENTS Widgets OpenGL Network REQUIRED)

message(STATUS "Found Qt5 version ${Qt5_VERSION}: ${Qt5_DIR}")

if (ZENO_WITH_PYTHON3)
    find_package(Python3 COMPONENTS Development REQUIRED)
endif()

#add_custom_target(
#    clean_resource_file
#    COMMAND ${CMAKE_COMMAND} -E remove -f ${PROJECT_BINARY_DIR}/ui/zenoedit/qrc_font.o
#    COMMAND ${CMAKE_COMMAND} -E remove -f ${PROJECT_BINARY_DIR}/ui/zenoedit/qrc_icons.o
#)

file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp *.ui)
file(GLOB_RECURSE resource CONFIGURE_DEPENDS *.qrc)
file(GLOB_RECURSE bigresource CONFIGURE_DEPENDS *icons.qrc *font.qrc)
FOREACH(item ${resource})
    IF(item IN_LIST bigresource)
        LIST(REMOVE_ITEM resource ${item})
    ENDIF(item IN_LIST bigresource)
ENDFOREACH(item)

qt5_add_big_resources(source ${bigresource})  # cihou msvc: fatal error C1060: compiler is out of heap space
qt5_add_resources(source ${resource})

add_executable(zenqt ${source} ${WIN32_RESOURCES})
# Add a dependency to ensure that the resource file is deleted before being regenerated

# add_dependencies(zenqt clean_resource_file)

set_property(TARGET zenqt PROPERTY AUTOUIC ON)
set_property(TARGET zenqt PROPERTY AUTOMOC ON)
set_property(TARGET zenqt PROPERTY AUTORCC ON)
set_property(TARGET zenqt PROPERTY WIN32_EXECUTABLE OFF)

if (ZENO_DEBUG_MSVC)
    zeno_dbg_msvc(zenqt)
endif()

#BEGIN PYTHON IMPORT
if (ZENO_WITH_PYTHON3)
message(STATUS "python3 libraries for zenqt: ${Python3_LIBRARIES}")
message(STATUS "python3 include directories for zenqt: ${Python3_INCLUDE_DIRS}")
target_link_libraries(zenqt PRIVATE ${Python3_LIBRARIES})
target_include_directories(zenqt PRIVATE ${Python3_INCLUDE_DIRS})
endif()
#ENDIF

if (NOT ZENO_MULTIPROCESS OR NOT ZENO_IPC_USE_TCP)
    find_package(Threads REQUIRED)
    target_link_libraries(zenqt PRIVATE Threads::Threads)
endif()
target_link_libraries(zenqt PRIVATE Qt5::Widgets Qt5::OpenGL Qt5::Network)

if (WIN32)
    target_link_libraries(zenqt PRIVATE Dbghelp Version)
    target_link_libraries(zenqt PRIVATE ws2_32.lib winmm.lib wldap32.lib Advapi32.lib Crypt32.lib)
endif()

target_link_libraries(zenqt PRIVATE zenoio zenovis)
target_link_libraries(zenqt PRIVATE zenocore)
target_include_directories(zenqt PRIVATE .)

if (ZENO_WITH_PYTHON3)
    target_compile_definitions(zenqt PRIVATE -DZENO_WITH_PYTHON3)
endif()


add_custom_command(TARGET zenqt PRE_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${PROJECT_SOURCE_DIR}/hdr ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hdr)