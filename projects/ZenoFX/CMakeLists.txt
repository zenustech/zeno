add_subdirectory(ZFX)

target_include_directories(zeno PRIVATE .)

target_link_libraries(zeno PUBLIC ZFX)
target_sources(zeno PRIVATE nw.cpp pw.cpp pnw.cpp ppw.cpp pew.cpp dbg_printf.h)

if (EXTENSION_zenvdb)
    option(ZENOFX_ENABLE_OPENVDB "Enable OpenVDB wrangler in ZenoFX" ON)
else()
    option(ZENOFX_ENABLE_OPENVDB "Enable OpenVDB wrangler in ZenoFX" OFF)
endif()
option(ZENOFX_ENABLE_LBVH "Enable Linear BVH wrangler in ZenoFX" OFF)

if (ZENOFX_ENABLE_OPENVDB)
    find_package(OpenVDB REQUIRED)
    target_sources(zeno PRIVATE vw.cpp)
    target_include_directories(zeno PRIVATE ../zenvdb/include)
    target_link_libraries(zeno PRIVATE OpenVDB::openvdb)
endif()

if (ZENOFX_ENABLE_LBVH)
    target_sources(zeno PRIVATE pnbvhw.cpp LinearBvh.cpp LinearBvh.h SpatialUtils.hpp)
endif()

find_package(OpenMP)
if (TARGET OpenMP::OpenMP_CXX)
    message(STATUS "found package: OpenMP::OpenMP_CXX")
    target_link_libraries(zeno PRIVATE OpenMP::OpenMP_CXX)
endif()
