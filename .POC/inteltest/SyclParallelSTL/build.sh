#! /bin/bash

function display_help() {

cat <<EOT

To use build.sh to compile SyclParallelSTL with ComputeCpp:

  ./build.sh [--no-download] "path/to/ComputeCpp"
  (the path to ComputeCpp can be relative)

  For example:
  ./build.sh /home/user/ComputeCpp


To use build.sh to compile SyclParallelSTL with triSYCL:

  ./build.sh [--no-download] --trisycl [-DTRISYCL_INCLUDE_DIR=path/to/triSYCL/include] [-DBOOST_COMPUTE_INCLUDE_DIR=path/to/boost/compute/include]

  For example (Ubuntu 16.04):
  ./build.sh --trisycl -DTRISYCL_INCLUDE_DIR=~/triSYCL/include -DBOOST_COMPUTE_INCLUDE_DIR=~/compute/include


The "--no-download" option is used to skip the cloning or pulling of
the GoogleTest git repository.

EOT
}


# Useless to go on when an error occurs
set -o errexit

# Minimal emergency case to display the help message whatever happens
trap display_help ERR

if [ $1 == "--no-download" ]; then
  NO_DOWNLOAD="echo Skipping"
  shift
else
  NO_DOWNLOAD=
fi

if [ $1 == "--trisycl" ]
then
  shift
  echo "build.sh entering mode: triSYCL"
  # Pass all the remaining arguments to CMake
  CMAKE_ARGS="$CMAKE_ARGS -DUSE_COMPUTECPP=OFF $@"
else
  echo "build.sh entering mode: ComputeCpp"
  CMAKE_ARGS="$CMAKE_ARGS -DCOMPUTECPP_PACKAGE_ROOT_DIR=$(readlink -f $1)"
  shift
fi

NPROC=$(nproc)

function install_gmock  {(
  REPO="https://github.com/google/googletest.git"
  mkdir -p external
  cd external
  if [ -d googletest ]
  then
    cd googletest
    $NO_DOWNLOAD git pull
  else
    $NO_DOWNLOAD git clone $REPO
    cd googletest
  fi
  cd googlemock/make
  make -j$NPROC
)}

function configure  {
  mkdir -p build && pushd build
  cmake .. $CMAKE_ARGS -DPARALLEL_STL_BENCHMARKS=ON
  popd
}

function mak  {
  pushd build && make -j$NPROC
  popd
}

function tst {
  pushd build/tests
  ctest -VV --timeout 60
  popd
}

function main {
  install_gmock
  configure
  mak
  tst
}

main
