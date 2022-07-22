add_library(zeno_Audio SHARED Audio.cpp)

if("${CMAKE_BUILD_TYPE}" STREQUAL "Release" AND MSVC)
    set_property(TARGET zeno_Audio PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
endif()

add_subdirectory(aquila)

target_link_libraries(zeno_Audio PRIVATE Aquila)
target_link_libraries(zeno_Audio PRIVATE zeno)
# target_link_libraries(zeno_Audio PRIVATE zeno_nodep)
