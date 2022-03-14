file(GLOB ARRAYOP_SOURCE *.cpp *.h)

add_library(zeno_arrayop SHARED ${ARRAYOP_SOURCE})
target_include_directories(zeno_arrayop PRIVATE .)
target_link_libraries(zeno_arrayop PRIVATE zeno)
