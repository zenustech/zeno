target_sources(zeno PRIVATE Audio.cpp AudioFile.h)

zeno_disable_warning(Audio.cpp)

add_subdirectory(aquila)

target_link_libraries(zeno PRIVATE Aquila)
