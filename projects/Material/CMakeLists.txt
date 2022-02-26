cmake_minimum_required(VERSION 3.10)

# ZENO extension must be a SHARED library with name starting with `zeno_`
# otherwise it won't be automcatically loaded on ZENO startup
add_library(zeno_Material SHARED
    MaterialOps.cpp
)

# Finally don't forget to link zeno.dll
target_link_libraries(zeno_Material PUBLIC zeno)
