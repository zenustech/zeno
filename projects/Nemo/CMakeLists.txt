file(GLOB NEMO_SOURCE *.cpp *.h)

target_sources(zeno PRIVATE ${NEMO_SOURCE})
target_include_directories(zeno PRIVATE .)


