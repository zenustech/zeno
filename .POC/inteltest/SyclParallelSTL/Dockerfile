FROM ubuntu:bionic

# Default values for the build
ARG git_branch
ARG git_slug
ARG c_compiler
ARG cxx_compiler
ARG impl
ARG target

RUN apt-get -yq update

# Utilities
RUN apt-get install -yq --allow-downgrades --allow-remove-essential            \
    --allow-change-held-packages git wget apt-utils cmake unzip                \
    libboost-all-dev software-properties-common 

RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test

RUN apt-get -yq update

# Clang 6.0
RUN if [ "${c_compiler}" = 'clang-6.0' ]; then apt-get install -yq             \
    --allow-downgrades --allow-remove-essential --allow-change-held-packages   \
     clang-6.0 libomp-dev; fi

# GCC 7
RUN if [ "${c_compiler}" = 'gcc-7' ]; then apt-get install -yq                 \
    --allow-downgrades --allow-remove-essential --allow-change-held-packages   \
    g++-7 gcc-7; fi

# OpenCL ICD Loader
RUN apt-get install -yq --allow-downgrades --allow-remove-essential           \
    --allow-change-held-packages ocl-icd-opencl-dev ocl-icd-dev opencl-headers

RUN git clone https://github.com/${git_slug}.git -b ${git_branch} /SyclParallelSTL

# Intel OpenCL Runtime
RUN if [ "${target}" = 'opencl' ]; then bash /SyclParallelSTL/.travis/install_intel_opencl.sh; fi

# SYCL
RUN if [ "${impl}" = 'triSYCL' ]; then cd /SyclParallelSTL && bash /SyclParallelSTL/.travis/build_triSYCL.sh; fi
RUN if [ "${impl}" = 'COMPUTECPP' ]; then cd /SyclParallelSTL && bash /SyclParallelSTL/.travis/build_computecpp.sh; fi

ENV CC=${c_compiler}
ENV CXX=${cxx_compiler}
ENV SYCL_IMPL=${impl}
ENV TARGET=${target}

CMD cd /SyclParallelSTL && \
    if [ "${SYCL_IMPL}" = 'triSYCL' ]; then \
      ./build.sh --trisycl -DTRISYCL_INCLUDE_DIR=/tmp/triSYCL-master/include; \
    elif [ "${SYCL_IMPL}" = 'COMPUTECPP' ]; then \
      if [ "${TARGET}" = 'host' ]; then \
        COMPUTECPP_TARGET="host" ./build.sh /tmp/ComputeCpp-latest; \
      else \
        /tmp/ComputeCpp-latest/bin/computecpp_info && \
        COMPUTECPP_TARGET="intel:cpu" ./build.sh /tmp/ComputeCpp-latest; \
      fi \
    else \
      echo "Unknown SYCL implementation ${SYCL_IMPL}"; return 1; \
    fi
