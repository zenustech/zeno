# https://stackoverflow.com/questions/4346412/how-to-prepend-all-filenames-on-the-list-with-common-path/31726284
function(list_transform_prepend var prefix)
    set(temp "")
    foreach(f ${${var}})
        list(APPEND temp "${prefix}${f}")
    endforeach()
    set(${var} "${temp}" PARENT_SCOPE)
endfunction()

