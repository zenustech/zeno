file(GLOB FBX_SOURCE *.cpp *.h)

target_sources(zeno PRIVATE ${FBX_SOURCE})
target_include_directories(zeno PRIVATE .)

set(MODELS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../FBX/dem-bones-bin/models)
if (NOT EXISTS MODELS_DIR)
    message(FATAL_ERROR "Please run: git submodule update --init --recursive")
endif()
zeno_add_asset_dir(${MODELS_DIR})
configure_file(zeno_SampleModel_config.h.in ${CMAKE_CURRENT_BINARY_DIR}/confinc/zeno_SampleModel_config.h @ONLY)
target_include_directories(zeno PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/confinc)
