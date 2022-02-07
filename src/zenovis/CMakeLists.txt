file(GLOB ZENVIS_SOURCE "*.cpp" "*.hpp" "*.h")

ADD_LIBRARY(zenovis OBJECT ${ZENVIS_SOURCE})
target_include_directories(zenovis PRIVATE . ..)
target_link_libraries(zenovis glad)
target_compile_definitions(zenovis PRIVATE ZENVIS_LIB)
