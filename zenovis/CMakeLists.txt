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
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/stbi/include>
    $<INSTALL_INTERFACE:include/Zeno/zenovis/include>)

if (ZENO_ENABLE_OPENEXR)
    # PENG-SENSEI: SO OPENEXR FXXK YOU!!!!
    find_package(IlmBase CONFIG)
    find_package(OpenEXR CONFIG)
    if (NOT TARGET OpenEXR::OpenEXR AND NOT TARGET OpenEXR::IlmIlmUtil AND NOT TARGET IlmBase::Half)
        message(WARNING "OpenEXR::OpenEXR or IlmBase::Half not found!")
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
    target_compile_definitions(zenovis PRIVATE -DZENO_ENABLE_OPENEXR)
endif()

if (ZENO_INSTALL_TARGET)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include DESTINATION include/Zeno/zenovis)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/glad/include DESTINATION include/Zeno/zenovis)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/stbi/include DESTINATION include/Zeno/zenovis)
    install(TARGETS zenovis EXPORT ZenoTargets)
endif()
