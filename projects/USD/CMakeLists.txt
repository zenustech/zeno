if(NOT ZENO_ENABLE_ZENOCORE)
    message(FATAL_ERROR "You MUST set ZENO_ENABLE_ZENOCORE=ON when USD enable")
endif()

file(GLOB USD_SOURCE USD.cpp USD.h)
add_library(zeno_usd SHARED ${USD_SOURCE})

target_include_directories(zeno_usd PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_include_directories(zeno_usd PUBLIC ../../zeno/include ../../zeno/tpls/include)
target_link_libraries(zeno_usd PUBLIC zenocore)

find_package(USD REQUIRED HINTS ${ZENO_USD_ROOT} NAMES pxr)
if(USD_FOUND)
    set(TBB_ROOT_DIR ${ZENO_USD_ROOT})
    # this is only for VS
    if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        set_property(TARGET zenoedit PROPERTY VS_DEBUGGER_ENVIRONMENT "path=$(path)" ${ZENO_USD_ROOT}/bin ${ZENO_USD_ROOT}/lib)
    endif()
    target_include_directories(zeno_usd PRIVATE ${PXR_INCLUDE_DIRS})
    target_link_directories(zeno_usd PRIVATE ${ZENO_USD_ROOT}/lib)
    target_link_libraries(zeno_usd PRIVATE tbb usd_ar usd_arch usd_kind usd_usdLux usd_usdGeom usd_pcp usd_usdUtils usd_usd usd_usdShade usd_sdf usd_ts usd_tf usd_vt usd_usdSkel usd_usdImaging usd_gf)
else()
    message(FATAL_ERROR "USD not found")
endif()

target_sources(zeno PRIVATE USD_Def.cpp)

target_include_directories(zeno PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(zeno PRIVATE zeno_usd)