cmake_minimum_required(VERSION 3.16)
project(cpp20test LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -fmodules)
set(PREBUILT_MODULE_PATH ${CMAKE_BINARY_DIR}/modules)

function(add_module name)
    file(MAKE_DIRECTORY ${PREBUILT_MODULE_PATH})
    add_custom_target(${name}.pcm
            COMMAND
                ${CMAKE_CXX_COMPILER}
                -std=c++20
                -stdlib=libc++
                ${CMAKE_CXX_FLAGS}
                -c
                ${CMAKE_CURRENT_SOURCE_DIR}/${ARGN}
                -Xclang -emit-module-interface
                -o ${PREBUILT_MODULE_PATH}/${name}.pcm
            )
endfunction()

add_compile_options(-fmodules)
add_compile_options(-stdlib=libc++)
add_compile_options(-fbuiltin-module-map)
add_compile_options(-fimplicit-module-maps)
add_compile_options(-fprebuilt-module-path=${PREBUILT_MODULE_PATH})


file(GLOB PROJECT_SOURCE *.cpp)
add_module(helloworld helloworld.cpp)
add_executable(main main.cpp helloworld.cpp)
add_dependencies(main helloworld.pcm)
