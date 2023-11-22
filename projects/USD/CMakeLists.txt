#
# XXX Please using '/' slash for file path on windows
#
# -DZENO_LIVESYNC=ON
#      sync server
#
# -DZENO_WITH_USD=ON
#
# -DZENO_USD_VERBOSE=ON
#      verbose message for dev
#
# -DZENO_USD_ROOT=""
#      usd building dir, e.g. c:/src/usd-build
#
# -DZENO_USD_DEV_ROOT=""
#      not used for production
#

file(GLOB USD_SOURCE *.cpp *.h)
zeno_disable_warning(${USD_SOURCE})
target_sources(zeno PRIVATE ${USD_SOURCE})
target_include_directories(zeno PRIVATE .)

if(DEFINED ZENO_USD_DEV_ROOT)
    file(GLOB DEV_SOURCE ${ZENO_USD_DEV_ROOT}/*.cpp ${ZENO_USD_DEV_ROOT}/*.h)
    message(STATUS "Dev Root ${ZENO_USD_DEV_ROOT} Get ${DEV_SOURCE}")
    target_include_directories(zeno PRIVATE ${ZENO_USD_DEV_ROOT})
    target_sources(zeno PRIVATE ${DEV_SOURCE})
else()
    file(GLOB DEV_SOURCE ${CMAKE_CURRENT_LIST_DIR}/dev/*.cpp ${CMAKE_CURRENT_LIST_DIR}/dev/*.h)
    target_include_directories(zeno PRIVATE ${CMAKE_CURRENT_LIST_DIR}/dev)
    target_sources(zeno PRIVATE ${DEV_SOURCE})
endif()

if(DEFINED ZENO_USD_ROOT)
    set(TBB_ROOT_DIR ${ZENO_USD_ROOT})
    find_package(USD REQUIRED HINTS ${ZENO_USD_ROOT} NAMES pxr)
    find_package(MaterialX REQUIRED HINTS ${ZENO_USD_ROOT})
    find_package(TBB REQUIRED COMPONENTS tbb)
    target_include_directories(zeno PRIVATE ${ZENO_USD_DEV_ROOT})
    if(USD_FOUND)
        target_include_directories(zeno PRIVATE ${PXR_INCLUDE_DIRS})
        target_link_libraries(zeno PRIVATE ar cameraUtil hd hf hgi hio usd usdGeom usdImaging usdImagingGL usdSkel)
    endif()
endif()

if(DEFINED ZENO_USD_VERBOSE)
    get_cmake_property(_variableNames VARIABLES)
    list (SORT _variableNames)
    foreach (_variableName ${_variableNames})
        message(STATUS "\t\t${_variableName}=${${_variableName}}")
    endforeach()
endif()