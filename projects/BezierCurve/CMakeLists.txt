
file(GLOB SRC_LIST *.cpp *.h)

target_include_directories(zeno PRIVATE .)
target_sources(zeno PRIVATE ${SRC_LIST})
