cmake_minimum_required(VERSION 3.18)    # support relative path, cuda_std_17, archive manipulation

message("===============================================================")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose Release or Debug")
endif()

# project meta
# ============
include("project_info.in")
project(${project_name} 
    VERSION     ${project_version}
    DESCRIPTION ${project_description}
    LANGUAGES   CXX)

if (MSVC)
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

list(APPEND CMAKE_MODULE_PATH /usr/local/lib/cmake/OpenVDB/)  # openvdb's default linux install location 
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMake ${CMAKE_CURRENT_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMake ${CMAKE_CURRENT_BINARY_DIR})

include(GNUInstallDirs)
# global rpath setup (before target creations)
# ============================================
# https://gitlab.kitware.com/cmake/community/-/wikis/doc/cmake/RPATH-handling#common-questions
# https://cmake.org/pipermail/cmake/2008-January/019290.html
if (NOT APPLE)
  set(CMAKE_INSTALL_RPATH $ORIGIN/../${CMAKE_INSTALL_LIBDIR})  # location of the binary requiring the dep
endif()

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}" isInstallRpathSystemDir)
if("${isInstallRpathSystemDir}" STREQUAL "-1")
  set(CMAKE_INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
endif("${isInstallRpathSystemDir}" STREQUAL "-1")
#[=[
file(RELATIVE_PATH relDir
  ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}
  ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_INSTALL_RPATH $ORIGIN $ORIGIN/${relDir})
]=]

# build options
# =============
option(CMAKE_EXPORT_COMPILE_COMMANDS "Display Compile Commands" 1)

option(ENABLE_PCH "Enable Precompiled Headers" OFF)
option(ZS_BUILD_SHARED_LIBS "Enable compilation of shared libraries" OFF)
set(ZS_ENABLE_OPENMP On CACHE BOOL "Enable openmp-related build")
set(ZS_ENABLE_CUDA On CACHE BOOL "Enable cuda-related build")
set(ZS_ENABLE_OPENVDB On CACHE BOOL "Enable openvdb-related build")

option(ZENSIM_INSTALL "Install targets" On)
option(ZENSIM_CORE_SHARED "Build dynamically linked version of the core library." ON)
option(ZENSIM_CORE_STATIC "Build statically linked version of the core library." ON)

# project base targets
# ====================

include(CMake/Utility.cmake)
include(CMake/ClangSupport.cmake)
include(CMake/Cache.cmake)
include(CMake/StaticAnalyzers.cmake)
include(CMake/Sanitizers.cmake)
include(CMake/CompilerWarnings.cmake)
# set_project_warnings(zsproj_deps)
include(CMake/Doxygen.cmake)
enable_doxygen()

# ---- dependencies ---- 
# ======================
add_library(zsproj_deps INTERFACE)
add_subdirectory(include/zensim/tpls) # zsproj_deps
enable_sanitizers(zsproj_deps)
if (ENABLE_PCH)
  # This sets a global PCH parameter, each project will build its own PCH, which
  # is a good idea if any #define's change
  # consider breaking this out per project as necessary 
  target_precompile_headers(zsproj_deps INTERFACE <vector> <string> <map> <utility> <cstdio>)
endif()

# ---- build setup ----- 
# ======================

if (ZS_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message("-- cuda-compiler " ${CMAKE_CUDA_COMPILER})
  else(CMAKE_CUDA_COMPILER)
    message(STATUS "No CUDA support")
    set(ZS_ENABLE_CUDA 0 FORCE)
  endif(CMAKE_CUDA_COMPILER)
  set(CUDA_FOUND ${CMAKE_CUDA_COMPILER})
endif(ZS_ENABLE_CUDA)

if (MSVC)
  target_compile_definitions(zsproj_deps INTERFACE NOMINMAX)
endif()
target_include_directories(zsproj_deps INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

add_library(zecomp_cxx_deps INTERFACE)
target_compile_features(zecomp_cxx_deps INTERFACE cxx_std_17)
target_link_libraries(zecomp_cxx_deps INTERFACE zsproj_deps)
target_compile_definitions(zecomp_cxx_deps
    INTERFACE   AssetDirPath="${PROJECT_SOURCE_DIR}/Assets/"
)

if (ZS_ENABLE_CUDA)
  add_library(zecomp_cuda_deps INTERFACE)
  target_compile_features(zecomp_cuda_deps INTERFACE cuda_std_17)
  target_link_libraries(zecomp_cuda_deps INTERFACE zsproj_deps)
endif(ZS_ENABLE_CUDA)

# ---- binaries ----
# ==================

# ---- Library ----
# =================
add_subdirectory(include/zensim)

# ---- Install ----
# =================
if (ZENSIM_INSTALL)

set(ZS_INSTALL_DEP_TARGETS zspartio)
if (ZS_ENABLE_OPENVDB)
  set(ZS_INSTALL_DEP_TARGETS ${ZS_INSTALL_DEP_TARGETS} zsopenvdb)
endif(ZS_ENABLE_OPENVDB)

set(INSTALL_TARGETS zecomp zpc zpctool zecomp_cxx_deps zsproj_deps ${ZS_INSTALL_DEP_TARGETS})
if (ZS_ENABLE_CUDA)
  set(INSTALL_TARGETS ${INSTALL_TARGETS} zecomp_cuda_deps)
endif(ZS_ENABLE_CUDA)

message("project name: ${project_name}")
message("package version: ${project_version}")
message("targets to install: ${INSTALL_TARGETS}")
message("runtime dst: ${CMAKE_INSTALL_BINDIR}/${project_name}")
message("archive dst: ${CMAKE_INSTALL_LIBDIR}/${project_name}")
message("include dst: ${CMAKE_INSTALL_INCLUDEDIR}")
message("install_rpath: $ORIGIN/../${CMAKE_INSTALL_LIBDIR}")

# libs, includes
install(TARGETS ${INSTALL_TARGETS}
  EXPORT ${project_name}-targets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/${project_name}
          COMPONENT ${project_name}_Runtime
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/${project_name}
          COMPONENT ${project_name}_Runtime
          NAMELINK_COMPONENT ${project_name}_Development
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/${project_name}
          COMPONENT ${project_name}_Development
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  # PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/zensim # this relates to maxos framework
)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/zensim/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/zensim
  PATTERN "*.h|*.hpp|*.tpp|*.cuh"
  PATTERN "tpls/*"
  # FILES_MATCHING PATTERN "*.h|*.hpp|*.tpp|*.cuh"
)

# config, version
include(CMakePackageConfigHelpers)
set(ZENSIM_CONFIG_VERSION_FILE "${CMAKE_CURRENT_BINARY_DIR}/${project_name}-config-version.cmake")
set(ZENSIM_CONFIG_FILE "${CMAKE_CURRENT_BINARY_DIR}/${project_name}-config.cmake")
write_basic_package_version_file(
  ${ZENSIM_CONFIG_VERSION_FILE}
  VERSION ${project_version}
  COMPATIBILITY AnyNewerVersion
)
set(cmakeModulesDir cmake)
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/CMake/${project_name}-config.cmake.in ${ZENSIM_CONFIG_FILE}
  PATH_VARS cmakeModulesDir
  #INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${project_name}
  INSTALL_DESTINATION share/${project_name}/cmake
)
install(FILES ${ZENSIM_CONFIG_FILE} ${ZENSIM_CONFIG_VERSION_FILE} 
  # DESTINATION ${CMAKE_INSTALL_SYSCONFDIR}/${project_name}/cmake
  DESTINATION share/${project_name}/cmake
)
# target file for build tree
export(EXPORT ${project_name}-targets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/cmake/${project_name}-targets.cmake
  NAMESPACE ${project_name}:: 
)

# target files export
install(EXPORT ${project_name}-targets
  DESTINATION share/${project_name}/cmake
  NAMESPACE ${project_name}:: 
  # DESTINATION ${CMAKE_INSTALL_SYSCONFDIR}/${project_name}/cmake
)

add_custom_target(uninstall
  COMMAND ${CMAKE_COMMAND} -P ${PROJECT_SOURCE_DIR}/CMake/Uninstall.cmake
)
endif(ZENSIM_INSTALL)