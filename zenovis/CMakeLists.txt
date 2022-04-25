option(ZENO_ENABLE_OPENEXR "Enable OpenEXR in ZENO for HDR images I/O" OFF)

enable_language(C)

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
    # https://github.com/AcademySoftwareFoundation/openvdb/blob/ac97c17ed88dcfb392a2262076a1c7b5056298a8/openvdb/openvdb/cmd/CMakeLists.txt
    # PENG-SENSEI: SO OPENEXR FXXK YOU!!!!
    find_package(Imath CONFIG)
    if (NOT TARGET Imath::Imath)
      find_package(IlmBase REQUIRED COMPONENTS Half Iex IlmThread Imath)
    endif()
    if (NOT TARGET Imath::Imath)
      find_package(OpenEXR REQUIRED COMPONENTS IlmImf)
    else()
      find_package(OpenEXR CONFIG REQUIRED)
    endif()
    target_link_libraries(zenovis PRIVATE
    # For Imath/OpenEXR v3.X
    $<TARGET_NAME_IF_EXISTS:Imath::Imath>
    $<TARGET_NAME_IF_EXISTS:OpenEXR::OpenEXR>
    $<TARGET_NAME_IF_EXISTS:OpenEXR::OpenEXRUtil>
    $<TARGET_NAME_IF_EXISTS:OpenEXR::IlmThread>
    $<TARGET_NAME_IF_EXISTS:OpenEXR::Iex>
    # For IlmBase/OpenEXR v2.X
    $<TARGET_NAME_IF_EXISTS:IlmBase::Half>
    $<TARGET_NAME_IF_EXISTS:OpenEXR::IlmImf>
    $<TARGET_NAME_IF_EXISTS:OpenEXR::IlmImfUtil>
    $<TARGET_NAME_IF_EXISTS:IlmBase::IlmThread>
    $<TARGET_NAME_IF_EXISTS:IlmBase::Iex>
    $<TARGET_NAME_IF_EXISTS:IlmBase::Imath>
    )
    target_compile_definitions(zenovis PRIVATE -DZENO_ENABLE_OPENEXR)
endif()

if (ZENO_INSTALL_TARGET)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include DESTINATION include/Zeno/zenovis)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/glad/include DESTINATION include/Zeno/zenovis)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/stbi/include DESTINATION include/Zeno/zenovis)
    install(TARGETS zenovis EXPORT ZenoTargets)
endif()
