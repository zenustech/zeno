# target_sources(zeno PRIVATE 
# test_NeighborSearch_BVH.cpp
# )

# target_compile_definitions(zeno PRIVATE
# PBD_TEST_PATH="projects/test/"
# )

# target_compile_definitions(zeno PRIVATE
# PBD_TEST_INPUT="projects/test/input_data"
# )

# target_compile_definitions(zeno PRIVATE
# PBD_TEST_OUTPUT="projects/test/"
# )

add_executable(test_PBDCloth test_PBDCloth.cpp)
target_link_libraries(test_PBDCloth PRIVATE legacyPBD)
