file(GLOB dirs LIST_DIRECTORIES TRUE *)

foreach (dir ${dirs})
    if (IS_DIRECTORY ${dir})
        cmake_path(GET dir FILENAME name)
        add_subdirectory(${name})
    endif()
endforeach()
