option(ZENO_ENABLE_OPENEXR "Enable OpenEXR in ZENO for HDR images I/O" OFF)

file(GLOB_RECURSE source CONFIGURE_DEPENDS include/*.h src/*.cpp)
file(GLOB_RECURSE glad_source CONFIGURE_DEPENDS glad/include/*.h glad/src/*.c)
file(GLOB_RECURSE stbi_source CONFIGURE_DEPENDS stbi/include/*.h stbi/src/*.c)

add_library(zenovis OBJECT ${source} ${glad_source} ${stbi_source})
target_link_libraries(zenovis PRIVATE ${CMAKE_DL_LIBS})
target_link_libraries(zenovis PUBLIC zeno)
target_include_directories(zenovis PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/glad/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/stbi/include>)

add_subdirectory(zhxxvis)
target_link_libraries(zenovis PRIVATE zhxxvis)

option(ZENO_ENABLE_OPTIX "Enable NVIDIA OptiX in ZENO for path tracing" OFF)
if (ZENO_ENABLE_OPTIX)
    if (NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/xinxinoptix/CMakeLists.txt)
        message(FATAL_ERROR "xinxinoptix submodule not found! Please run: git submodule update --init --recursive")
    endif()
add_subdirectory(xinxinoptix)
target_link_libraries(zenovis PRIVATE xinxinoptix)
endif()

if (ZENO_ENABLE_OPENEXR)
    option(ZENO_SYSTEM_OPENEXR "Use OpenEXR found in system instead of submodule" OFF)
    if (ZENO_SYSTEM_OPENEXR)
        # PENG-SENSEI: SO OPENEXR FXXK YOU!!!!
        find_package(IlmBase CONFIG)
        find_package(OpenEXR CONFIG)
        if (NOT TARGET OpenEXR::OpenEXR
                AND NOT TARGET IlmBase::Iex
                AND NOT TARGET IlmBase::Half
                AND NOT TARGET IlmBase::Imath
                AND NOT TARGET IlmBase::IexMath
                AND NOT TARGET OpenEXR::IlmImf
                AND NOT TARGET OpenEXR::IlmImfUtil
                AND NOT TARGET OpenEXR::IlmImfConfig
                )
            message(FATAL_ERROR "OpenEXR or IlmBase not found! Please run: apt install -y libopenexr-dev")
        endif()
        target_link_libraries(zenovis PRIVATE
            $<TARGET_NAME_IF_EXISTS:IlmBase::Iex>
            $<TARGET_NAME_IF_EXISTS:IlmBase::Half>
            $<TARGET_NAME_IF_EXISTS:IlmBase::Imath>
            $<TARGET_NAME_IF_EXISTS:IlmBase::IexMath>
            $<TARGET_NAME_IF_EXISTS:OpenEXR::OpenEXR>
            $<TARGET_NAME_IF_EXISTS:OpenEXR::IlmImf>
            $<TARGET_NAME_IF_EXISTS:OpenEXR::IlmImfUtil>
            $<TARGET_NAME_IF_EXISTS:OpenEXR::IlmImfConfig>
        )
    else()
        if (NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/openexr/CMakeLists.txt)
            message(FATAL_ERROR "openexr submodule not found! Please run: git submodule update --init --recursive")
        endif()
        set(OPENEXR_FORCE_INTERNAL_IMATH OFF CACHE BOOL "FXXK YOU IMATH")
        add_subdirectory(openexr)
        target_link_libraries(zenovis PRIVATE OpenEXR)
        target_include_directories(zenovis PRIVATE openexr/src/lib/OpenEXR)
    endif()
    target_compile_definitions(zenovis PRIVATE -DZENO_ENABLE_OPENEXR)
endif()

#if (ZENO_INSTALL_TARGET)
    #install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include DESTINATION include/Zeno/zenovis)
    #install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/glad/include DESTINATION include/Zeno/zenovis)
    #install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/stbi/include DESTINATION include/Zeno/zenovis)
    #install(TARGETS zenovis EXPORT ZenoTargets)
#endif()
