add_library(ZenoModule_primitive OBJECT)

file(GLOB_RECURSE source CONFIGURE_DEPENDS
    #zeno/*.h zeno/*.cpp
    zeno2/*.h zeno2/*.cpp
    )
target_sources(ZenoModule_primitive PRIVATE ${source})
