cmake_minimum_required(VERSION 3.15)

cmake_host_system_information(RESULT OS_TYPE QUERY DISTRIB_PRETTY_NAME)

option(TRY_SYSTEM_LIB OFF)

if((TRY_SYSTEM_LIB) AND (OS_TYPE MATCHES "Arch Linux"))
  message(STATUS "Using USD install from pacman")
  target_include_directories(zeno PRIVATE /usr/include/python3.11)
  find_package(pxr REQUIRED)
  find_package(MaterialX REQUIRED)
  find_package(Imath REQUIRED)
  find_package(OpenGL REQUIRED)
  find_package(X11 REQUIRED)
  set(linklist tbb usd_ms MaterialXCore X11 Imath::ImathConfig OpenGL)
  target_link_libraries(zeno PRIVATE ${linklist}) 
endif()

file(GLOB USD_SOURCE *.cpp *.h)
zeno_disable_warning(${USD_SOURCE})
target_sources(zeno PRIVATE ${USD_SOURCE})
target_include_directories(zeno PRIVATE .)



if(DEFINED ZENO_USD_ROOT)
    set(TBB_ROOT_DIR ${ZENO_USD_ROOT})
    find_package(USD REQUIRED HINTS ${ZENO_USD_ROOT} NAMES pxr)
    # find_package(MaterialX REQUIRED HINTS ${ZENO_USD_ROOT})
    # find_package(TBB REQUIRED COMPONENTS tbb)
    target_include_directories(zeno PRIVATE ${ZENO_USD_DEV_ROOT})
    if(USD_FOUND)
        # this is only for VS
        set_property(TARGET zenoedit PROPERTY VS_DEBUGGER_ENVIRONMENT "path=$(path)" ${ZENO_USD_ROOT}/bin ${ZENO_USD_ROOT}/lib)

        target_include_directories(zeno PRIVATE ${PXR_INCLUDE_DIRS})
        target_link_directories(zeno PRIVATE ${ZENO_USD_ROOT}/lib)
        target_link_libraries(zeno PRIVATE tbb usd_ar usd_arch usd_kind usd_usdLux usd_usdGeom usd_pcp usd_usdUtils usd_usd usd_usdShade usd_sdf usd_ts usd_tf usd_vt usd_usdSkel usd_usdImaging usd_gf)
    endif()
endif()
