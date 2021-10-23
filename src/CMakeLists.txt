add_library(zeno OBJECT)
target_include_directories(zeno PUBLIC .)

add_subdirectory(zeno/ztd)
add_subdirectory(zeno/dop)
add_subdirectory(zeno/types)
#add_subdirectory(zeno/legacy)
#add_subdirectory(zeno/modules)
