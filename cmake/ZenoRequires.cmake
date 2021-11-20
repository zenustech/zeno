include(cmake/ZPM.cmake)

zpm_requires(
#pkgs/BoostBuilder
#pkgs/zlib
#pkgs/c-blosc,-DBUILD_TESTS:BOOL=OFF
#pkgs/tbb,-DTBB_BUILD_TESTS:BOOL=OFF
#pkgs/openvdb
#pkgs/eigen
pkgs/fmt,-DFMT_DOC:BOOL=OFF,-DFMT_TEST:BOOL=OFF
pkgs/spdlog,-DSPDLOG_FMT_EXTERNAL:BOOL=ON,-DSPDLOG_BUILD_EXAMPLE:BOOL=OFF
)
