file(GLOB dirs LIST_DIRECTORIES TRUE *)

foreach (dir ${dirs})
    if (IS_DIRECTORY ${dir})
        get_filename_component(name ${dir} NAME)
        add_subdirectory(${name})
    endif()
endforeach()
