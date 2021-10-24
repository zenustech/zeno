target_link_libraries(zs PUBLIC glm)

file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp)
target_sources(zs PRIVATE ${source})

find_package(OpenMP)
if (TARGET OpenMP::OpenMP_CXX)
    message("-- Found: OpenMP::OpenMP_CXX")
    target_link_libraries(zs PUBLIC OpenMP::OpenMP_CXX)
else()
    message("-- Not Found: OpenMP::OpenMP_CXX")
endif()
