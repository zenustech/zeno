cmake_minimum_required(VERSION 3.16)
project(cpp20test LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
#set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_CXX_EXTENSIONS OFF)
set(PREBUILT_MODULE_PATH ${CMAKE_BINARY_DIR}/modules)

function(target_as_module name)
    file(MAKE_DIRECTORY ${PREBUILT_MODULE_PATH})

    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_options(${name} PUBLIC -fmodules)
        target_compile_options(${name} PUBLIC -stdlib=libc++)
        target_compile_options(${name} PUBLIC -fbuiltin-module-map)
        target_compile_options(${name} PUBLIC -fimplicit-module-maps)
        target_compile_options(${name} PUBLIC -fprebuilt-module-path=${PREBUILT_MODULE_PATH})
        target_compile_options(${name} PRIVATE -fmodule-file=${PREBUILT_MODULE_PATH}/${name}.pcm)
        add_custom_target(${name}.pcm
                COMMAND
                    ${CMAKE_CXX_COMPILER}
                    -std=c++20
                    -stdlib=libc++
                    -fmodules
                    ${CMAKE_CXX_FLAGS}
                    -c
                    ${CMAKE_CURRENT_SOURCE_DIR}/${ARGN}
                    -Xclang -emit-module-interface
                    -o ${PREBUILT_MODULE_PATH}/${name}.pcm
                )
        add_dependencies(${name} ${name}.pcm)

    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(${name} PUBLIC -fmodules-ts)
        target_compile_options(${name} PUBLIC -fbuiltin-module-map)
        target_compile_options(${name} PRIVATE -fmodule-file=${PREBUILT_MODULE_PATH}/${name}.pcm)
        add_custom_target(${name}.pcm
                COMMAND
                    ${CMAKE_CXX_COMPILER}
                    -std=c++20
                    -fmodules-ts
                    ${CMAKE_CXX_FLAGS}
                    -c
                    ${CMAKE_CURRENT_SOURCE_DIR}/${ARGN}
                    -emit-module-interface
                    -o ${PREBUILT_MODULE_PATH}/${name}.gcm
                )
        add_dependencies(${name} ${name}.pcm)

    else()
        message(FATAL_ERROR "Compiler ID `${CMAKE_CXX_COMPILER_ID}` is not supported for using C++20 modules now.")
    endif()
endfunction()


add_library(helloworld OBJECT helloworld.cpp)
target_as_module(helloworld helloworld.cc)
add_executable(main main.cpp)
target_link_libraries(main helloworld)
