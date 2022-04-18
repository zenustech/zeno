file(GLOB PROJECT_SOURCE include/*/*.h *.cpp)

if (UNIX)
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} /usr/local/lib/cmake/OpenVDB)
endif()

####################################################
# FUN FACT: DID U KNOW THAT MICROSOFT LOVES LINUX? #
####################################################
if (MSVC)  # must use this to make msvc linker happy
    add_compile_options(
            $<$<CONFIG:>:/bigobj>
            $<$<CONFIG:Debug>:/bigobj>
            $<$<CONFIG:Release>:/bigobj>
    )  # 节数超过对象文件格式限制: 请使用 /bigobj 进行编译 
endif()
####################################################
# FUN FACT: DID U KNOW THAT MICROSOFT BUYS GITHUB? #
####################################################

target_sources(zeno PRIVATE ${PROJECT_SOURCE})
target_include_directories(zeno PRIVATE ../oldzenbase/include)

target_include_directories(zeno PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/Zeno/zeno>)

if (ZENO_INSTALL_TARGET)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include DESTINATION include/Zeno/zeno)
endif()

find_package(OpenMP)
if (TARGET OpenMP::OpenMP_CXX)
    message("found package: OpenMP::OpenMP_CXX")
    target_link_libraries(zeno PRIVATE OpenMP::OpenMP_CXX)
endif()

find_package(TBB REQUIRED COMPONENTS tbb tbbmalloc)
find_package(OpenVDB REQUIRED)
target_link_libraries(zeno PRIVATE OpenVDB::openvdb TBB::tbbmalloc TBB::tbb)
if (WIN32)
  target_compile_options(zeno PRIVATE -DNOMINMAX -D_USE_MATH_DEFINES)
endif()
