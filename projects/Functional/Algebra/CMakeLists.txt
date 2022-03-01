cmake_minimum_required(VERSION 3.16)
set(PROJECT_NAME IBFE)
project(${PROJECT_NAME})

# add subdirectories
set(MY_DIRS loguru AlgebraSolver)
set(SOURCE_FILE)
set(MY_SOURCES)

foreach(DIR ${MY_DIRS})
        add_subdirectory(${DIR})
        message("subdirectory : " ${DIR}  "\n")
        # Add files in ${DIR}
        foreach(SOURCE_FILE ${SOURCES})
                list(APPEND MY_SOURCES ${DIR}/${SOURCE_FILE})
                message("source file : " ${DIR}/${SOURCE_FILE}  "\n")
        endforeach()
endforeach()

# GPU_lib is compiled seperately.
add_subdirectory(GPULIB)

# add executable
add_library(${PROJECT_NAME} OBJECT ${MY_SOURCES})

# find and use packages
find_package(DOLFIN REQUIRED)
include(${DOLFIN_USE_FILE})

# include headers
target_include_directories(${PROJECT_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

# link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE pthread dl)
# target_link_libraries(${PROJECT_NAME} PRIVATE dolfin mshr)
target_link_libraries(${PROJECT_NAME} PUBLIC gpu_lib2)

# add compiling flags
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall -g -std=c++17")


# add compiling definitions
# target_compile_definitions(${PROJECT_NAME} PUBLIC DEBUG)

# add tests
# add_subdirectory(test)
