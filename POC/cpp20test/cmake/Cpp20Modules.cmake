if (NOT PREBUILT_MODULE_PATH)
    set(PREBUILT_MODULE_PATH ${CMAKE_BINARY_DIR}/modules)
endif()

function(target_module_interface name)
    file(MAKE_DIRECTORY ${PREBUILT_MODULE_PATH})

    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set(stdlib libstdc++)
        target_link_libraries(${name} PUBLIC c dl stdc++ m)
        target_compile_options(${name} PUBLIC -fmodules)
        target_compile_options(${name} PUBLIC -stdlib=${stdlib})
        target_compile_options(${name} PUBLIC -fbuiltin-module-map)
        target_compile_options(${name} PUBLIC -fimplicit-module-maps)
        target_compile_options(${name} PUBLIC -fprebuilt-module-path=${PREBUILT_MODULE_PATH})
        target_compile_options(${name} PRIVATE -fmodule-file=${PREBUILT_MODULE_PATH}/${name}.pcm)
        add_custom_target(${name}.pcm
                COMMAND
                    ${CMAKE_CXX_COMPILER}
                    -std=c++20
                    -stdlib=${stdlib}
                    -DCPP20_MODULE_HEADER_${name}=1
                    -fmodules
                    ${CMAKE_CXX_FLAGS}
                    -c
                    ${CMAKE_CURRENT_SOURCE_DIR}/${ARGN}
                    -Xclang -emit-module-interface
                    -o ${PREBUILT_MODULE_PATH}/${name}.pcm
                )
        add_dependencies(${name} ${name}.pcm)

    else()
        message(FATAL_ERROR "Compiler ID `${CMAKE_CXX_COMPILER_ID}` is not supported for using C++20 modules now. Please consider use Clang instead.")
    endif()
endfunction()


function(add_module name)
    add_library(${name} OBJECT ${name}.cc)
    target_module_interface(${name} ${name}.hh)
endfunction()
