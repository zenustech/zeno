file(GLOB_RECURSE source CONFIGURE_DEPENDS src/*.h src/*.cpp)
add_library(zeno_core OBJECT ${source})

target_include_directories(zeno_core PUBLIC src)
target_include_directories(zeno_core PUBLIC ../3rdparty/include)

target_include_directories(zeno_core PUBLIC ../3rdparty/mods/mcut/include)
target_link_libraries(zeno_core PUBLIC mcut)

find_package(OpenMP)
if (TARGET OpenMP::OpenMP_CXX)
    message(STATUS "Found OpenMP: ${OpenMP_DIR}")
    target_link_libraries(zeno_core PUBLIC OpenMP::OpenMP_CXX)
else()
    message(WARNING "Not Found OpenMP")
endif()

if (ZENO_WITH_LEGACY)
    message(STATUS "Building Zeno with Legacy Nodes")
    add_subdirectory(legacy)
endif()

if (ZENO_WITH_BACKWARD)
    message(STATUS "Building Zeno with Stack Traceback")
    target_sources(zeno_core PRIVATE ${BACKWARD_ENABLE})
    add_backward(zeno_core)
endif()
