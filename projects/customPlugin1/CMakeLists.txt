add_library(customPlugin1 SHARED plugindef.h pluginnode.cpp)

target_link_libraries(customPlugin1 PUBLIC zenocore)

if (ZENO_WITH_REFLECTOR)
    include(${CMAKE_SOURCE_DIR}/reflect/cmake/ReflectionUtils.cmake)

    set(GENERATE_ON_BUILD_DIR ON)
    set(REFLECT_TEMPLATE_INCLUDE "#include <zeno/types/ObjectDef.h>")
    set(INJA_TEMPLATE_DIR_PATH  ${CMAKE_BINARY_DIR}/intermediate)
    #头文件包含目录
    set(CUSTOM_HEADER_INCLUDE_DIR ${CMAKE_BINARY_DIR}/intermediate)
    #由于这个template的字符数过大导致MSVC编译期截断，需要分发template，以读写的方式加载
    file(COPY ${PROJECT_SOURCE_DIR}/reflect/src/template/reflected_type_register.inja DESTINATION ${INJA_TEMPLATE_DIR_PATH})

    make_absolute_paths(REFLECTION_HEADERS 
        plugindef.h
        pluginnode.cpp
    )
    #送入Generator并生成
    zeno_declare_reflection_support(customPlugin1 "${REFLECTION_HEADERS}" custom_plugin1)
endif()