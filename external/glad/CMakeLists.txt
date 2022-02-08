# pip install glad
# python -m glad --out-path . --generator c --api gl=3.2 --profile core
add_library(glad STATIC src/glad.c)
target_include_directories(glad PUBLIC include)
target_link_libraries(glad PUBLIC ${CMAKE_DL_LIBS})
