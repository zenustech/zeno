file(GLOB_RECURSE source CONFIGURE_DEPENDS include/*.h include/*.cpp)

add_library(zenovis OBJECT ${source})
target_link_libraries(zenovis PUBLIC glad zenoinclude zenostbi zenocore)
target_include_directories(zenovis PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/Zeno/zenovis>)

if (ZENO_INSTALL_TARGET)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include DESTINATION include/Zeno/zenovis)
endif()
