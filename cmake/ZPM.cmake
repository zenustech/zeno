cmake_minimum_required(VERSION 3.18)
project(ZPMPackageRoot)

if (NOT CMAKE_BUILD_TYPE)
    message(FATAL_ERROR "CMAKE_BUILD_TYPE not set, please specify it with, e.g. -DCMAKE_BUILD_TYPE=Release")
endif()
if (NOT ZPM_INSTALL_PREFIX)
    set(ZPM_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/.venv)
    message(STATUS "Setting ZPM_INSTALL_PREFIX to [${ZPM_INSTALL_PREFIX}] by default")
endif()

message(STATUS "ZPM_INSTALL_PREFIX: [${ZPM_INSTALL_PREFIX}]")
message(STATUS "PROJECT_BINARY_DIR: [${PROJECT_BINARY_DIR}]")
message(STATUS "CMAKE_BUILD_TYPE: [${CMAKE_BUILD_TYPE}]")
message(STATUS "CMAKE_COMMAND: [${CMAKE_COMMAND}]")
message(STATUS "CMAKE_GENERATOR: [${CMAKE_GENERATOR}]")

function (_zpm_install pkg_desc)
    string(REPLACE "," ";" pkg_args ${pkg_desc})
    list(POP_FRONT pkg_args pkg_path)

    execute_process(COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --yellow "======================================")
    message(STATUS "ZPM installing package [${pkg_desc}]")
    execute_process(COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --yellow "======================================")

    file(REMOVE_RECURSE ${PROJECT_BINARY_DIR}/zpm_pkg_build/${pkg_path})

    execute_process(
        COMMAND ${CMAKE_COMMAND}
        -G ${CMAKE_GENERATOR}
        -Wno-dev -S ${pkg_path}
        -B ${PROJECT_BINARY_DIR}/zpm_pkg_build/${pkg_path}
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_INSTALL_PREFIX=${ZPM_INSTALL_PREFIX}
        -DCMAKE_FIND_USE_CMAKE_PATH:BOOL=TRUE
        -DCMAKE_FIND_USE_CMAKE_SYSTEM_PATH:BOOL=FALSE
        -DCMAKE_PREFIX_PATH=${ZPM_INSTALL_PREFIX}
        ${pkg_args}
        RESULT_VARIABLE ret
        )
    if (NOT "${ret}" STREQUAL "0")
        message(FATAL_ERROR "Configuring [${pkg_path}] failed with [${ret}]")
    endif()

    include(ProcessorCount)
    ProcessorCount(cpu_count)
    if(NOT cpu_count EQUAL 0)
      set(parallel_flag --parallel ${cpu_count})
    else()
      set(parallel_flag)
    endif()
    execute_process(
        COMMAND ${CMAKE_COMMAND}
        --build ${PROJECT_BINARY_DIR}/zpm_pkg_build/${pkg_path}
        --config ${CMAKE_BUILD_TYPE}
        ${parallel_flag}
        RESULT_VARIABLE ret
        )
    if (NOT "${ret}" STREQUAL "0")
        message(FATAL_ERROR "Building [${pkg_path}] failed with [${ret}]")
    endif()

    if (NOT "${pkg_path}" MATCHES "^BoostBuilder$")
        execute_process(
            COMMAND ${CMAKE_COMMAND}
            --build ${PROJECT_BINARY_DIR}/zpm_pkg_build/${pkg_path}
            --config ${CMAKE_BUILD_TYPE}
            --target install
            RESULT_VARIABLE ret
            )
        if (NOT "${ret}" STREQUAL "0")
            message(FATAL_ERROR "Installing [${pkg_path}] failed with [${ret}]")
        endif()
    else()
        message(STATUS "Skipping install for [${pkg_path}] as it have no install target")
    endif()

    execute_process(COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --cyan "======================================")
    message(STATUS "ZPM installed package: [${pkg_desc}]")
    execute_process(COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --cyan "======================================")
endfunction()

function(zpm_requires)
    set(lock_path ${ZPM_INSTALL_PREFIX}/zpm.lock)
    if (EXISTS ${lock_path})
        file(READ ${lock_path} content)
    else()
        set(content ZPM_INVALID_CONTENT)
    endif()
    if (NOT "${content}" STREQUAL "${ARGN}")
        message(STATUS "ZPM installing requirements: [${ARGN}]")
        foreach (pkg_desc ${ARGN})
            _zpm_install(${pkg_desc})
        endforeach()
        file(WRITE ${lock_path} "${ARGN}")
        message(STATUS "ZPM successfully installed: [${ARGN}]")
    endif()
    set(CMAKE_INSTALL_PREFIX ${ZPM_INSTALL_PREFIX} CACHE PATH "For ZPM virtual environment" FORCE)
    set(CMAKE_PREFIX_PATH ${ZPM_INSTALL_PREFIX} CACHE PATH "For ZPM virtual environment" FORCE)
    set(CMAKE_FIND_USE_CMAKE_SYSTEM_PATH FALSE CACHE BOOL "For ZPM virtual environment" FORCE)
    set(CMAKE_FIND_USE_CMAKE_PATH TRUE CACHE BOOL "For ZPM virtual environment" FORCE)
endfunction()
