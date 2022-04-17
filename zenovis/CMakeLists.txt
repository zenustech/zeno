file(GLOB_RECURSE source CONFIGURE_DEPENDS include/*.h include/*.cpp)

add_library(zenovis ${ZENO_LIBTYPE} ${source})
target_link_libraries(zenovis PUBLIC glad zenoinclude zenostbi zenocore)
target_include_directories(zenovis PUBLIC .)
