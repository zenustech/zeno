include(cmake/ZPM.cmake)

zpm_requires(
#3rdparty/BoostBuilder
#3rdparty/zlib
#3rdparty/c-blosc,-DBUILD_TESTS:BOOL=OFF
#3rdparty/tbb,-DTBB_BUILD_TESTS:BOOL=OFF
#3rdparty/openvdb
#3rdparty/eigen
3rdparty/fmt,-DFMT_DOC:BOOL=OFF,-DFMT_TEST:BOOL=OFF
3rdparty/spdlog,-DSPDLOG_FMT_EXTERNAL:BOOL=ON,-DSPDLOG_BUILD_EXAMPLE:BOOL=OFF
)
