# pip install glad
# python -m glad --out-path . --generator c --api gl=4.1
add_library(glad STATIC src/glad.c)
target_include_directories(glad PUBLIC include)
if (UNIX)
    target_link_libraries(glad PUBLIC dl)
endif()
