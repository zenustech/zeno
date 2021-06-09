# - Try to find Half
# Once done, this will define
#
#  HALF_FOUND - system has Half
#  HALF_INCLUDE_DIRS - the Half include directories
#  HALF_LIBRARIES - link these to use libHalf

find_path(HALF_ROOT_DIR
    NAMES include/OpenEXR/half.h
    HINTS ENV HT
)

find_library(HALF_LIBRARIES
    NAMES Half libHalf
    HINTS ${HALF_ROOT_DIR}/lib
    ENV HDSO
)

find_path(HALF_INCLUDE_DIRS
    NAMES OpenEXR/half.h
    HINTS ${HALF_ROOT_DIR}/include
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Half DEFAULT_MSG
    HALF_LIBRARIES
    HALF_INCLUDE_DIRS
)

mark_as_advanced(
    HALF_ROOT_DIR
    HALF_LIBRARIES
    HALF_INCLUDE_DIRS
)
