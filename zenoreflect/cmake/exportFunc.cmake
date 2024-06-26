set(RELFECTION_GENERATION_ROOT_TARGET _Reflection_ROOT CACHE INTERNAL "Reflection generator dependencies for all targets")
add_custom_target(${RELFECTION_GENERATION_ROOT_TARGET})

macro(make_absolute_paths out_var)
    set(result_list)
    foreach(path ${ARGN})
        get_filename_component(abs_path "${path}" ABSOLUTE)
        list(APPEND result_list "${abs_path}")
    endforeach()
    set(${out_var} "${result_list}")
endmacro()

set(INTERMEDIATE_FILE_BASE_DIR "${CMAKE_BINARY_DIR}/intermediate")

function(zeno_declare_reflection_support target reflection_headers)
    # Call this function after all target source has been added
    set(splitor ",")

    set(INTERMEDIATE_FILE_DIR "${INTERMEDIATE_FILE_BASE_DIR}/${target}")
    set(INTERMEDIATE_ALL_IN_ONE_FILE "${INTERMEDIATE_FILE_DIR}/${target}.generated.cpp")
    file(WRITE "${INTERMEDIATE_ALL_IN_ONE_FILE}" "// TBD by reflection generator\n")
    target_sources(${target} PRIVATE "${INTERMEDIATE_ALL_IN_ONE_FILE}")

    # Input sources
    get_target_property(REFLECTION_GENERATION_SOURCE ${target} SOURCES)
    get_target_property(REFLECTION_GENERATION_SOURCE_DIR ${target} SOURCE_DIR)
    list(LENGTH REFLECTION_GENERATION_SOURCE source_files_length)
    if (source_files_length EQUAL 0)
        message(WARNING "There is not source files found in target ${target}, check your calling timing")
    endif()
    set(source_paths_value ${REFLECTION_GENERATION_SOURCE})
    list(JOIN reflection_headers ${splitor} source_paths_string)

    # Include dirs
    set(INCLUDE_DIRS $<LIST:REMOVE_DUPLICATES,$<TARGET_PROPERTY:${target},INCLUDE_DIRECTORIES>>)
    # Obtain compiler built-in include paths
    list(JOIN CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES "," SYSTEM_IMPLICIT_INCLUDE_DIRS)

    set(REFLECTION_GENERATION_TARGET _internal_${target}_reflect_generation)

    add_custom_target(${REFLECTION_GENERATION_TARGET}
        WORKING_DIRECTORY
            ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND 
            $<TARGET_FILE:ZenoReflect::generator> --include_dirs=\"$<JOIN:${INCLUDE_DIRS},${splitor}>,${SYSTEM_IMPLICIT_INCLUDE_DIRS}\" --pre_include_header="${LIBREFLECT_PCH_PATH}" --input_source=\"${source_paths_string}\" --header_output="${ZENO_REFLECTION_GENERATED_HEADERS_DIR}" --stdc++=${CMAKE_CXX_STANDARD} $<IF:$<CONFIG:Debug>,-v,> --generated_source_path="${INTERMEDIATE_ALL_IN_ONE_FILE}"
        SOURCES 
            ${reflection_headers} 
        COMMENT 
            "Generating reflection information for ${target}..."
    )
    add_dependencies(${RELFECTION_GENERATION_ROOT_TARGET} ${REFLECTION_GENERATION_TARGET})
    add_dependencies(${target} ${RELFECTION_GENERATION_ROOT_TARGET})

    target_link_libraries(${target} PUBLIC ZenoReflect::libreflect ZenoReflect::libgenerated)
endfunction()
