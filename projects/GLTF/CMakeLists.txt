file(GLOB GLTF_SOURCE *.cpp *.h)

target_sources(zeno PRIVATE ${GLTF_SOURCE})
target_include_directories(zeno PRIVATE .)

find_package(draco CONFIG REQUIRED)
target_link_libraries(zeno PRIVATE draco::draco)

