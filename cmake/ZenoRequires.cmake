include(cmake/ZPM.cmake)

zpm_requires(
    #3rdparty/pkgs/BoostBuilder
    #3rdparty/pkgs/zlib
    #3rdparty/pkgs/c-blosc,-DBUILD_TESTS:BOOL=OFF
    3rdparty/pkgs/tbb,-DTBB_BUILD_TESTS:BOOL=OFF
    #3rdparty/pkgs/openvdb
    #3rdparty/pkgs/eigen
    3rdparty/pkgs/fmt,-DFMT_DOC:BOOL=OFF,-DFMT_TEST:BOOL=OFF
    3rdparty/pkgs/spdlog,-DSPDLOG_FMT_EXTERNAL:BOOL=ON,-DSPDLOG_BUILD_EXAMPLE:BOOL=OFF
)
