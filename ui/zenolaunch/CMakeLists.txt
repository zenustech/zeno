find_package(Qt5 COMPONENTS Widgets OpenGL Network REQUIRED)
message(STATUS "Found Qt5 version ${Qt5_VERSION}: ${Qt5_DIR}")

option(ZENO_MULTIPROCESS "Enable multiprocessing for ZENO" ON)

file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp *.ui * ../zenoedit/launch/runnermain.cpp ../zenoedit/launch/startup.cpp)

# add_subdirectory(zenoedit.out EXCLUDE_FROM_ALL ../zenoedit/main.cpp ../zenoedit/CMakeLists.txt)
# add_subdirectory(../zenoedit)
add_executable(zenolaunch ${source})

set_property(TARGET zenolaunch PROPERTY AUTOUIC ON)
set_property(TARGET zenolaunch PROPERTY AUTOMOC ON)
set_property(TARGET zenolaunch PROPERTY AUTORCC ON)
set_property(TARGET zenolaunch PROPERTY WIN32_EXECUTABLE OFF)

find_package(Threads REQUIRED)
target_link_libraries(zenolaunch PRIVATE Threads::Threads)
target_link_libraries(zenolaunch PRIVATE Qt5::Widgets Qt5::OpenGL Qt5::Network)
target_link_libraries(zenolaunch PRIVATE zenoui zenoio zenovis zenomodel)
target_link_libraries(zenolaunch PRIVATE zeno)
target_include_directories(zenolaunch PRIVATE . .. ../zenoui ../zenoedit ../include ../zenomodel/include)
target_include_directories(zenolaunch PRIVATE ${Qt5Widgets_PRIVATE_INCLUDE_DIRS})

add_compile_definitions(ZENO_MULTIPROCESS ZENO_IPC_USE_TCP)