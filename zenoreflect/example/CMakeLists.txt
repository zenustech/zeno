option(REFLECT_BUILD_EXAMPLE "Build reflection example target" ON)

if (REFLECT_BUILD_EXAMPLE)

    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)

    make_absolute_paths(REFLECTION_HEADERS 
        include/data.h
        include/test.h
        include/print.h
    ) 

    function(add_single_file_test_target name)
        set(target_name Reflect-Tests-${name})
        add_executable(${target_name}
            "src/${name}.cpp"
        )
        target_include_directories(${target_name} PUBLIC
            include
        )
        target_compile_definitions(${target_name} PRIVATE
            REFLECT_TEST_NAME="${name}"
        )
        zeno_declare_reflection_support(${target_name} "${REFLECTION_HEADERS}")
    endfunction(add_single_file_test_target)

    add_single_file_test_target(example)
    add_single_file_test_target(field_visit)
    add_single_file_test_target(any_with_ptr)

endif()
