file(GLOB ZENVIS_SOURCE "zenovis/*.cpp" "zenovis/*.hpp" "zenovis/*.h")

ADD_LIBRARY(zenovis OBJECT ${ZENVIS_SOURCE})
target_link_libraries(zenovis PRIVATE glad zenoinclude zenodes)
target_include_directories(zenovis PRIVATE . ..)
