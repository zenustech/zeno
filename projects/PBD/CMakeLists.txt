target_sources(zeno PRIVATE PBD.cpp PBD.h)

#if("${CMAKE_BUILD_TYPE}" STREQUAL "Release" AND MSVC)
    #set_property(TARGET zeno_Audio PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
#endif()


target_link_libraries(zeno PRIVATE PBD)
