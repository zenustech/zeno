add_subdirectory(ZFX)

add_library(zeno_ZenoFX SHARED)
target_include_directories(zeno_ZenoFX PRIVATE .)

target_link_libraries(zeno_ZenoFX PUBLIC zeno ZFX)
target_link_libraries(zeno_ZenoFX PRIVATE zeno_nodep)
target_sources(zeno_ZenoFX PRIVATE nw.cpp pw.cpp pnw.cpp ppw.cpp pew.cpp dbg_printf.h)

if (EXTENSION_zenvdb)
    option(ZENOFX_ENABLE_OPENVDB "Enable OpenVDB wrangler in ZenoFX" ON)
    option(ZENOFX_ENABLE_LBVH "Enable Linear BVH wrangler in ZenoFX" ON)
else()
    option(ZENOFX_ENABLE_OPENVDB "Enable OpenVDB wrangler in ZenoFX" OFF)
    option(ZENOFX_ENABLE_LBVH "Enable Linear BVH wrangler in ZenoFX" OFF)
endif()

if (ZENOFX_ENABLE_OPENVDB)
    find_package(OpenVDB REQUIRED)
    target_sources(zeno_ZenoFX PRIVATE vw.cpp)
    target_include_directories(zeno_ZenoFX PRIVATE ../zenvdb/include)
    target_link_libraries(zeno_ZenoFX PRIVATE OpenVDB::openvdb)
endif()

if (ZENOFX_ENABLE_LBVH)
    target_sources(zeno_ZenoFX PRIVATE pnbvhw.cpp LinearBvh.cpp LinearBvh.h SpatialUtils.hpp)
endif()

find_package(OpenMP)
if (TARGET OpenMP::OpenMP_CXX)
    message("found package: OpenMP::OpenMP_CXX")
    target_link_libraries(zeno_ZenoFX PRIVATE OpenMP::OpenMP_CXX)
endif()
