file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp *.ui)

add_executable(autotest ${source}) 

set_property(TARGET autotest PROPERTY WIN32_EXECUTABLE OFF)

target_link_libraries(autotest PRIVATE zenoui zenoio zenomodel zeno)
target_include_directories(autotest PRIVATE . .. ../zenoui ../include)
