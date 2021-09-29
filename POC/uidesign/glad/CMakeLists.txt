# pip install glad
# python -m glad --out-path . --generator c --api gl=2.0
add_library(glad STATIC src/glad.c)
target_include_directories(glad PUBLIC include)
