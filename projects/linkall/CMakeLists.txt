set(CMAKE_LINK_WHAT_YOU_USE TRUE)

add_library(zeno_linkall SHARED dummy.cpp)

foreach (name IN ITEMS ${EXTENSIONS_LIST})
    if (TARGET zeno_${name})
        target_link_libraries(zeno_linkall PRIVATE zeno_${name})
    endif()
endforeach()
