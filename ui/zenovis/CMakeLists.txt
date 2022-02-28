file(GLOB ZENVIS_SOURCE "*.cpp" "*.hpp" "*.h")

ADD_LIBRARY(zenovis OBJECT ${ZENVIS_SOURCE})
target_link_libraries(zenovis PRIVATE glad zenoinclude zenodes)
target_include_directories(zenovis PRIVATE . ..)
