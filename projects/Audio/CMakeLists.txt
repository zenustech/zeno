target_sources(zeno PRIVATE Audio.cpp AudioFile.h)

#if("${CMAKE_BUILD_TYPE}" STREQUAL "Release" AND MSVC)
    #set_property(TARGET zeno_Audio PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
#endif()

add_subdirectory(aquila)

target_link_libraries(zeno PRIVATE Aquila)
