# - Try to find tbb
# Once done, this will define
#
#  TBB_FOUND - system has tbb
#  TBB_INCLUDE_DIRS - the tbb include directories
#  TBB_LIBRARIES - link these to use libtbb

find_path(TBB_ROOT_DIR
    NAMES include/tbb/tbb.h
    HINTS
    ENV HT
)

find_library(TBB_LIBRARIES
    NAMES tbb libtbb
    HINTS
    ${TBB_ROOT_DIR}/lib
    ENV HDSO
)

find_library(TBB_MALLOC
    NAMES tbbmalloc libtbbmalloc
    HINTS
    ${TBB_ROOT_DIR}/lib
    ENV HDSO
)

find_library(TBB_MALLOC_PROXY
    NAMES tbbmalloc_proxy libtbbmalloc_proxy
    HINTS
    ${TBB_ROOT_DIR}/lib
    ENV HDSO
)

find_path(TBB_INCLUDE_DIRS
    NAMES tbb/tbb.h
    HINTS ${TBB_ROOT_DIR}/include
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TBB DEFAULT_MSG
    TBB_LIBRARIES
    TBB_INCLUDE_DIRS
)

mark_as_advanced(
    TBB_ROOT_DIR
    TBB_LIBRARIES
    TBB_MALLOC
    TBB_MALLOC_PROXY
    TBB_INCLUDE_DIRS
)
