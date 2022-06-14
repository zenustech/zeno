find_package(Qt5 COMPONENTS Widgets OpenGL REQUIRED)
message(STATUS "Found Qt5 version ${Qt5_VERSION}: ${Qt5_DIR}")

file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp *.ui ../zenoedit/*)
file(GLOB_RECURSE resource CONFIGURE_DEPENDS *.qrc)
qt5_add_big_resources(source ${resource})  # cihou msvc: fatal error C1060: compiler is out of heap space

# add_subdirectory(zenoedit.out EXCLUDE_FROM_ALL ../zenoedit/main.cpp ../zenoedit/CMakeLists.txt)
# add_subdirectory(../zenoedit)
list(REMOVE_ITEM source  ${CMAKE_CURRENT_SOURCE_DIR}/../zenoedit/main.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../zenoedit/CMakeLists.txt)
add_executable(zenoplayer ${source})

set_property(TARGET zenoplayer PROPERTY AUTOUIC ON)
set_property(TARGET zenoplayer PROPERTY AUTOMOC ON)
set_property(TARGET zenoplayer PROPERTY AUTORCC ON)
set_property(TARGET zenoplayer PROPERTY WIN32_EXECUTABLE OFF)

find_package(Threads REQUIRED)
target_link_libraries(zenoplayer PRIVATE Threads::Threads)
target_link_libraries(zenoplayer PRIVATE Qt5::Widgets Qt5::OpenGL)
target_link_libraries(zenoplayer PRIVATE zenoui zenoio zenovis)
target_link_libraries(zenoplayer PRIVATE zeno)
target_include_directories(zenoplayer PRIVATE . .. ../zenoui ../zenoedit)
target_include_directories(zenoplayer PRIVATE ${Qt5Widgets_PRIVATE_INCLUDE_DIRS})


if (ZENO_INSTALL_TARGET)
    install(TARGETS zenoplayer EXPORT ZenoTargets)
endif()
