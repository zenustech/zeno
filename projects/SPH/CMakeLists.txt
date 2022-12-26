# set(SPH_LIBS_ONLY ON)
add_subdirectory(SPlisHSPlasH)

target_link_libraries(zeno PRIVATE SPlisHSPlasH)

target_sources(zeno PRIVATE
    nosys/DFSPH.cpp
)

target_include_directories(zeno
PRIVATE
    .
)