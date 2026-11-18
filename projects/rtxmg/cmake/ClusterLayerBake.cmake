# Keep robust-predicate types and floating-point options in the offline baker;
# the renderer does not link these libraries. Sources are pinned submodules.
set(_zeno_layer_supported OFF)
if(CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|MSVC)$")
  set(_zeno_layer_supported ON)
endif()
option(ZENO_CLUSTER_LAYER_BAKE "Build scene-aware layer preservation into the asset baker" ${_zeno_layer_supported})
if(NOT ZENO_CLUSTER_LAYER_BAKE)
  return()
endif()
if(NOT _zeno_layer_supported)
  message(FATAL_ERROR "Layer-preserving baking needs GCC or MSVC floating-point rounding support; configure ZENO_CLUSTER_LAYER_BAKE=OFF for the other asset tools")
endif()
foreach(_dep fast-envelope nfg indirect-predicates)
  if(NOT EXISTS "${ZENO_CLUSTER_ROOT}/projects/${_dep}/.git")
    message(FATAL_ERROR "Initialize the bake dependency: git submodule update --init projects/${_dep}")
  endif()
endforeach()
find_package(Eigen3 CONFIG REQUIRED)
set(_envelope_src "${ZENO_CLUSTER_ROOT}/projects/fast-envelope/src/fastenvelope")
add_library(zeno_layer_envelope STATIC
  "${_envelope_src}/AABB.cpp" "${_envelope_src}/FastEnvelope.cpp"
  "${_envelope_src}/FastEnvelope2D.cpp" "${_envelope_src}/Morton.cpp"
  "${_envelope_src}/common_algorithms.cpp" "${_envelope_src}/indirectPredicates/ip_filtered_ex.cpp")
target_include_directories(zeno_layer_envelope PUBLIC
  "${ZENO_CLUSTER_ROOT}/projects/fast-envelope/src"
  "${ZENO_CLUSTER_ROOT}/projects/nfg/include"
  "${ZENO_CLUSTER_ROOT}/projects/indirect-predicates/include")
target_link_libraries(zeno_layer_envelope PUBLIC Eigen3::Eigen)
target_compile_features(zeno_layer_envelope PUBLIC cxx_std_20)
target_compile_definitions(zeno_layer_envelope PUBLIC NFG_IP_ERROR_THROWS NOMINMAX)
if(MSVC)
  target_compile_options(zeno_layer_envelope PUBLIC /fp:strict)
else()
  target_compile_options(zeno_layer_envelope PUBLIC -frounding-math -ffp-contract=off)
endif()
add_library(zeno_cluster_layer_bake STATIC "${CMAKE_CURRENT_LIST_DIR}/../ClusterLayerBake.cpp")
target_link_libraries(zeno_cluster_layer_bake PUBLIC zeno_cluster_cache PRIVATE zeno_layer_envelope)
# Source's UTF-8 path interface is shared with the C++17 cache library. Keep
# that interface while compiling the predicate implementation as C++20.
if(MSVC)
  target_compile_options(zeno_cluster_layer_bake PRIVATE /Zc:char8_t-)
else()
  target_compile_options(zeno_cluster_layer_bake PRIVATE -fno-char8_t)
endif()
target_link_libraries(zeno_cluster_bake PRIVATE zeno_cluster_layer_bake)
target_compile_definitions(zeno_cluster_bake PRIVATE ZENO_CLUSTER_LAYER_BAKE=1)
