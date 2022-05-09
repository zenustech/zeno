cmake_minimum_required(VERSION 3.3)
project(lbfgspp VERSION 0.1.1 LANGUAGES CXX)

# + ----------------- +
# | BUILDING SETTINGS |
# + ----------------- +

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)

# + ----------------- +
# | COMPILATION FLAGS |
# + ----------------- +

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-Wall WALL_SUPPORTED)
if(${WALL_SUPPORTED})
  add_compile_options(-Wall)
endif()

# + --------------- +
# | LBFGSpp LIBRARY |
# + --------------- +

add_library(lbfgspp INTERFACE)

# + -------- +
# | INCLUDES |
# + -------- +

target_include_directories(lbfgspp INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include>
)

# + ----------------------- +
# | FIND EXTERNAL LIBRARIES |
# + ----------------------- +

find_package(Eigen3 3.0 REQUIRED)
target_link_libraries(lbfgspp INTERFACE Eigen3::Eigen)
message("-- Eigen3 version: " ${EIGEN3_VERSION_STRING})

# + ------------ +
# | INSTALLATION |
# + ------------ +

# Copy headers folder
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
  DESTINATION include
)
# Create an export set
install(TARGETS lbfgspp EXPORT lbfgsppTargets)

include(CMakePackageConfigHelpers)
# Version file
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/lbfgspp-config-version.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)
# Config file
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/lbfgspp-config.cmake
  INSTALL_DESTINATION lib/cmake/lbfgspp
)
# Targets files
export(
  EXPORT lbfgsppTargets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/lbfgspp-targets.cmake
)
install(
  EXPORT lbfgsppTargets
  FILE lbfgspp-targets.cmake
  DESTINATION lib/cmake/lbfgspp
)
install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/lbfgspp-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/lbfgspp-config-version.cmake
  DESTINATION lib/cmake/lbfgspp
)
