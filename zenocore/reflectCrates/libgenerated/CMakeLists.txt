
add_library(zenocorelibgenerated INTERFACE)
target_include_directories(zenocorelibgenerated INTERFACE include)

set(ZENO_REFLECTION_GENERATED_HEADERS_DIR "${CMAKE_CURRENT_LIST_DIR}/include" CACHE INTERNAL "Reflection generated header")
set(ZENO_REFLECTION_GENERATED_HEADER "${ZENO_REFLECTION_GENERATED_HEADERS_DIR}/generated_templates.hpp" CACHE INTERNAL "Reflection generated header")
