file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp *.ui)

find_package(Qt5 COMPONENTS Widgets OpenGL REQUIRED)

add_executable(autotest ${source}) 

set_property(TARGET autotest PROPERTY WIN32_EXECUTABLE OFF)

set_property(TARGET autotest PROPERTY AUTOUIC ON)
set_property(TARGET autotest PROPERTY AUTOMOC ON)
set_property(TARGET autotest PROPERTY AUTORCC ON)

target_link_libraries(autotest PUBLIC zenoapi zenomodel zenoio)
target_link_libraries(autotest PRIVATE Qt5::Widgets Qt5::OpenGL)
target_include_directories(autotest PRIVATE . .. ../zenoui ../include)
