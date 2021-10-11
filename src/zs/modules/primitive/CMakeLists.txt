target_link_libraries(zs PUBLIC glm)

file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp)
target_sources(zs PRIVATE ${source})
