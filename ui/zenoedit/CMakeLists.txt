if (ZENO_MULTIPROCESS)
    find_package(Qt5 COMPONENTS Widgets OpenGL Network REQUIRED)
else()
    find_package(Qt5 COMPONENTS Widgets OpenGL REQUIRED)
endif()
message(STATUS "Found Qt5 version ${Qt5_VERSION}: ${Qt5_DIR}")

file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp *.ui)
file(GLOB_RECURSE resource CONFIGURE_DEPENDS *.qrc)
qt5_add_big_resources(source ${resource})  # cihou msvc: fatal error C1060: compiler is out of heap space

add_executable(zenoedit ${source}) 

set_property(TARGET zenoedit PROPERTY AUTOUIC ON)
set_property(TARGET zenoedit PROPERTY AUTOMOC ON)
set_property(TARGET zenoedit PROPERTY AUTORCC ON)
set_property(TARGET zenoedit PROPERTY WIN32_EXECUTABLE OFF)

find_package(Threads REQUIRED)
target_link_libraries(zenoedit PRIVATE Threads::Threads)
add_definitions(-DMULTIPROC_BASE_TCP)  #multiprocess base tcp.
if (ZENO_MULTIPROCESS)
    target_link_libraries(zenoedit PRIVATE Qt5::Widgets Qt5::OpenGL Qt5::Network)
else()
    target_link_libraries(zenoedit PRIVATE Qt5::Widgets Qt5::OpenGL)
endif()
target_link_libraries(zenoedit PRIVATE zenoui zenoio zenovis)
target_link_libraries(zenoedit PRIVATE zeno)
target_include_directories(zenoedit PRIVATE . .. ../zenoui)
target_include_directories(zenoedit PRIVATE ${Qt5Widgets_PRIVATE_INCLUDE_DIRS})
if (ZENO_MULTIPROCESS)
    target_compile_definitions(zenoedit PRIVATE -DZENO_MULTIPROCESS)
endif()

if (ZENO_INSTALL_TARGET)
    install(TARGETS zenoedit EXPORT ZenoTargets)
endif()
