file(GLOB ZENOVIS_SOURCE "zenovis/*.cpp" "zenovis/*.hpp" "zenovis/*.h")

ADD_LIBRARY(zenovis OBJECT ${ZENOVIS_SOURCE})
target_link_libraries(zenovis PRIVATE glad zenoinclude zenodes)
target_include_directories(zenovis PUBLIC .)
