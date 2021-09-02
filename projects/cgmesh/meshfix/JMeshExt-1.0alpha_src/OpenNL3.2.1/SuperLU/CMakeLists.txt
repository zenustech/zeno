######################################################################
#
# CMakeLists.txt for SUPERLU
#
######################################################################

# Required version
cmake_minimum_required(VERSION 2.8.12)

# Project Version	
project(SuperLU NONE)
set(VERSION_MAJOR "5")
set(VERSION_MINOR "2")
set(VERSION_BugFix "2")
set(PROJECT_VERSION ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_BugFix})

######################################################################
#
# IDEAS: xSDK standards module
MESSAGE("\nProcess XSDK defaults ...")
# SET(USE_XSDK_DEFAULTS_DEFAULT TRUE) # Set to false if desired
INCLUDE("cmake/XSDKDefaults.cmake")
INCLUDE(CTest)

######################################################################

######################################################################
#
# Usual initialization stuff
#
######################################################################
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)    ## ????
set(CMAKE_INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/lib")

#---- For shared library

# use, i.e. don't skip the full RPATH for the build tree
SET(CMAKE_SKIP_BUILD_RPATH  FALSE)

# when building, don't use the install RPATH already
# (but later on when installing)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE) 

# the RPATH to be used when installing
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
#----

if (BUILD_SHARED_LIBS)
  message("-- SuperLU will be built as a shared library.")
  set(PROJECT_NAME_LIB_EXPORT libsuperlu.so)
else()
  message("-- SuperLU will be built as a static library.")
  set(PROJECT_NAME_LIB_EXPORT libsuperlu.a)
endif()

enable_language(C)
if (XSDK_ENABLE_Fortran)
  enable_language (Fortran)
  set(NOFORTRAN FALSE)
endif()
set(SUPERLU_VERSION "${PROJECT_VERSION}")
set(SUPERLU_REV "${PROJECT_REV}")

# The XSDK standard does not allow using internally built BLAS
if (USE_XSDK_DEFAULTS)
  set(enable_blaslib_DEFAULT OFF)
else()
  set(enable_blaslib_DEFAULT ON)
endif()

if (NOT CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX /usr/local)
endif()

# setup options
option(enable_internal_blaslib  "Build the CBLAS library" ${enable_blaslib_DEFAULT})
option(enable_matlabmex "Build the Matlab mex library" OFF)
option(enable_tests     "Build tests" ON)
option(enable_doc       "Build doxygen documentation" OFF)
option(enable_single    "Enable single precision library" ON)
option(enable_double    "Enable double precision library" ON)
option(enable_complex   "Enable complex precision library" ON)
option(enable_complex16 "Enable complex16 precision library" ON)
# option(enable_examples  "Build examples" ON)

# setup required compiler defines and options.
## add_definitions(-DPRNTlevel=0 -DAdd_)
## get_directory_property( DirDefs COMPILE_DEFINITIONS )
set(CMAKE_C_FLAGS "-DPRNTlevel=0 -DDEBUGlevel=0 ${CMAKE_C_FLAGS}")
set(CMAKE_C_FLAGS_RELEASE "-O3" CACHE STRING "")

######################################################################
#
# Find packages
#
######################################################################
#
#--------------------- BLAS ---------------------
if(NOT enable_internal_blaslib)
  if (TPL_BLAS_LIBRARIES)
    set(BLAS_FOUND TRUE)
  else()
    find_package(BLAS)
    if (BLAS_FOUND)
      set(TPL_BLAS_LIBRARIES "${BLAS_LIBRARIES}" CACHE FILEPATH
        "Set from FindBLAS.cmake BLAS_LIBRARIES." FORCE)
    endif()
  endif()
endif()

if(BLAS_FOUND)
  message("-- Using TPL_BLAS_LIBRARIES='${TPL_BLAS_LIBRARIES}'")
  set(CMAKE_C_FLAGS "-DUSE_VENDOR_BLAS ${CMAKE_C_FLAGS}")
  set(BLAS_LIB ${TPL_BLAS_LIBRARIES})
  # fix up BLAS library name
  string (REPLACE ";" " " BLAS_LIB_STR "${BLAS_LIB}")
  set(BLAS_LIB_EXPORT ${BLAS_LIB_STR})
else()
  message("-- Did not find or specify BLAS so configure to build internal CBLAS ...")
  add_subdirectory(CBLAS)
  set(BLAS_LIB blas)
  if (BUILD_SHARED_LIBS)  # export to be referenced by downstream makefile
      set(BLAS_LIB_EXPORT ${CMAKE_INSTALL_PREFIX}/CBLAS/libblas.so)
  else()
      set(BLAS_LIB_EXPORT ${CMAKE_INSTALL_PREFIX}/CBLAS/libblas.a)
  endif()
endif()

######################################################################
#
# Add subdirectories
#
######################################################################

add_subdirectory(SRC)

if(enable_matlabmex)
  add_subdirectory(MATLAB)
endif()

if(enable_tests)
  enable_testing()
  add_subdirectory(TESTING)
endif()

if(enable_doc)
  message(FATAL_ERROR "Documentation build requested but not implemented.")
  #implement doxygen
endif()

# file(WRITE "make.defs" "# can be exposed to users"
#  ${CMAKE_C_COMPILER}  )
# configure_file(${CMAKE_SOURCE_DIR}/make.inc.in ${CMAKE_SOURCE_DIR}/make.inc)
configure_file(${SuperLU_SOURCE_DIR}/make.inc.in ${SuperLU_SOURCE_DIR}/make.inc)
 
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/superlu.pc.in ${CMAKE_CURRENT_BINARY_DIR}/superlu.pc @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/superlu.pc
	DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
