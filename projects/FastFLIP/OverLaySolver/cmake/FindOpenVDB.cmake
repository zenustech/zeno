# - Try to find openvdb
# Once done, this will define
#
#  OPENVDB_FOUND - system has openvdb
#  OPENVDB_INCLUDE_DIRS - the openvdb include directories
#  OPENVDB_LIBRARIES - link these to use libopenvdb

find_library(OPENVDB_LIBRARIES
    NAMES openvdb openvdb_sesi
    HINTS ENV HDSO
)

find_path(OPENVDB_INCLUDE_DIRS
    NAMES openvdb/openvdb.h
    HINTS $ENV{HT}/include
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenVDB DEFAULT_MSG
    OPENVDB_LIBRARIES
    OPENVDB_INCLUDE_DIRS
)

mark_as_advanced(
    OPENVDB_LIBRARIES
    OPENVDB_INCLUDE_DIRS
)
