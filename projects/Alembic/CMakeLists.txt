cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 17)
project(zeno_Alembic)

include (common)

find_package(Alembic REQUIRED)

file(GLOB NOSYS_SOURCE *.cpp *.h)

add_library(zeno_Alembic SHARED ${NOSYS_SOURCE})

#simd compiler flags
if (MSVC)
    target_compile_options(zeno_Alembic PRIVATE "/arch:AVX")
else()
    target_compile_options(zeno_Alembic PRIVATE -mavx -mfma)
endif()

target_include_directories(zeno_Alembic PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(zeno_Alembic PRIVATE Alembic::Alembic)
target_link_libraries(zeno_Alembic PRIVATE zeno)
