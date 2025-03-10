find_package(Qt5 COMPONENTS Widgets OpenGL Network REQUIRED)

message(STATUS "Found Qt5 version ${Qt5_VERSION}: ${Qt5_DIR}")

file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp)
file(GLOB_RECURSE resource CONFIGURE_DEPENDS *.qrc)
file(GLOB_RECURSE bigresource CONFIGURE_DEPENDS *zsg.qrc)
FOREACH(item ${resource})
    IF(item IN_LIST bigresource)
        LIST(REMOVE_ITEM resource ${item})
    ENDIF(item IN_LIST bigresource)
ENDFOREACH(item)

qt5_add_big_resources(source ${bigresource})  # cihou msvc: fatal error C1060: compiler is out of heap space
qt5_add_resources(source ${resource})

#may be a shared lib
add_executable(zensolver ${source})

set_property(TARGET zensolver PROPERTY AUTOUIC ON)
set_property(TARGET zensolver PROPERTY AUTOMOC ON)
set_property(TARGET zensolver PROPERTY AUTORCC ON)
set_property(TARGET zensolver PROPERTY WIN32_EXECUTABLE OFF)

if (ZENO_DEBUG_MSVC)
    zeno_dbg_msvc(zensolver)
endif()

if (NOT ZENO_MULTIPROCESS OR NOT ZENO_IPC_USE_TCP)
    find_package(Threads REQUIRED)
    target_link_libraries(zensolver PRIVATE Threads::Threads)
endif()
target_link_libraries(zensolver PRIVATE Qt5::Core)

if (WIN32)
    target_link_libraries(zensolver PRIVATE Dbghelp Version)
    target_link_libraries(zensolver PRIVATE ws2_32.lib winmm.lib wldap32.lib Advapi32.lib Crypt32.lib)
endif()
target_link_libraries(zensolver PRIVATE zeno zenoio zenomodel)
target_include_directories(zensolver PRIVATE . ./include ../ui  ../ui/include)
target_include_directories(zensolver PRIVATE ${Qt5Widgets_PRIVATE_INCLUDE_DIRS})

target_compile_definitions(zensolver PRIVATE -DZENO_MULTIPROCESS)
target_compile_definitions(zensolver PRIVATE -DZENO_IPC_USE_TCP)