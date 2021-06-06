if (LAPACK_LIBRARIES)
  set (LAPACK_FIND_QUIETLY TRUE)
endif (LAPACK_LIBRARIES)

find_library (LAPACK_LIBRARIES
  lapack
  HINTS
  LAPACK_LIB
  $ENV{LAPACK_LIB}
  ${LIB_INSTALL_DIR}
  )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LAPACK DEFAULT_MSG LAPACK_LIBRARIES)

mark_as_advanced (LAPACK_LIBRARIES)
