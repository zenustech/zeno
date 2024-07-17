
add_library(libreflect SHARED
    src/reflect.cpp
    src/registry.cpp
    src/typeinfo.cpp
    src/type.cpp

    src/impl/memory.cpp
    src/impl/exit.cpp
    src/impl/metadata.cpp
)
add_library(ZenoReflect::libreflect ALIAS libreflect)
target_include_directories(libreflect PUBLIC include)
target_include_directories(libreflect PRIVATE include/reflect)
target_compile_definitions(libreflect PRIVATE "LIBREFLECT_EXPORTS=1")

set(PRE_INCLUDE_HEADER "${CMAKE_CURRENT_LIST_DIR}/include/reflect/core.hpp")
# target_precompile_headers(libreflect PUBLIC ${PRE_INCLUDE_HEADER})
get_target_property(LIBREFLECT_INCLUDE_DIRS libreflect INCLUDE_DIRECTORIES)
set(LIBREFLECT_INCLUDE_DIRS ${LIBREFLECT_INCLUDE_DIRS} CACHE INTERNAL "LIBREFLECT_INCLUDE_DIRS")
set(LIBREFLECT_PCH_PATH ${PRE_INCLUDE_HEADER} CACHE INTERNAL "Reflection PCH path")
