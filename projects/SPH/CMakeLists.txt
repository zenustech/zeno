option(SPH_LIBS_ONLY "Build SPH lib only, no demo" OFF)
add_subdirectory(SPlisHSPlasH)

target_link_libraries(zeno PRIVATE SPlisHSPlasH)

target_include_directories(zeno
PRIVATE
    SPlisHSPlasH
)

target_sources(zeno PRIVATE
    nosys/DFSPH.cpp
)
