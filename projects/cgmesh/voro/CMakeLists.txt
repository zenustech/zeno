cmake_minimum_required(VERSION 3.10)

project(voro++ VERSION 0.4.6 LANGUAGES CXX)
set(SOVERSION "0")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CXX_FLAGS)
  #release comes with -O3 by default
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CXX_FLAGS)

########################################################################
# User input options                                                   #
########################################################################
option(VORO_BUILD_SHARED_LIBS "Build shared libs" ON)
include(GNUInstallDirs)

########################################################################
#Find external packages
########################################################################
find_package(Doxygen)

######################################
# Include the following subdirectory # 
######################################

file(GLOB VORO_SOURCES src/*.cc)
file(GLOB NOT_VORO_SOURCES src/v_base_wl.cc src/cmd_line.cc src/voro++.cc)
list(REMOVE_ITEM VORO_SOURCES ${NOT_VORO_SOURCES})
add_library(voro++ ${VORO_SOURCES})
set_target_properties(voro++ PROPERTIES 
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/src"
  SOVERSION ${SOVERSION})
install(TARGETS voro++ EXPORT VORO_Targets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
#for voto++.hh
target_include_directories(voro++ PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/inc/voro++)
target_include_directories(voro++ PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/inc> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

file(GLOB_RECURSE VORO_HEADERS src/inc/voro++/voro++.hh)
install(FILES ${VORO_HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES ${CMAKE_SOURCE_DIR}/man/voro++.1 DESTINATION ${CMAKE_INSTALL_MANDIR}/man1)
# no external deps for we can use target file as config file
install(EXPORT VORO_Targets FILE VOROConfig.cmake NAMESPACE VORO:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/VORO)
include(CMakePackageConfigHelpers)
write_basic_package_version_file("VOROConfigVersion.cmake" VERSION ${PROJECT_VERSION} COMPATIBILITY ExactVersion)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/VOROConfigVersion.cmake" DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/VORO)

if (DOXYGEN_FOUND)
  add_custom_target(doxygen COMMAND ${DOXYGEN_EXECUTABLE} src/Doxyfile 
    COMMENT "Build doxygen documentation")
endif (DOXYGEN_FOUND)
