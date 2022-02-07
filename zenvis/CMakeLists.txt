file(GLOB ZENVIS_SOURCE "*.cpp" "*.hpp" "*.h")

ADD_LIBRARY(zenvis STATIC ${ZENVIS_SOURCE})
target_include_directories(zenvis PRIVATE .)
target_include_directories(zenvis PRIVATE ${PROJECT_SOURCE_DIR}/zeno)
target_link_libraries(zenvis glad)
target_compile_definitions(zenvis PRIVATE ZENVIS_LIB)
