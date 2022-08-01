file(GLOB_RECURSE ZENVIS_SOURCE CONFIGURE_DEPENDS "*.cpp" "*.hpp" "*.h")
target_sources(zenovis PRIVATE ${ZENVIS_SOURCE})
target_include_directories(zenovis PRIVATE .)
target_include_directories(zenovis PRIVATE ../glad/include ../stbi/include)
