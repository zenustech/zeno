cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 17)
project(zeno_Alembic)

file(GLOB ABC_SOURCE *.cpp *.h)

add_library(zeno_Alembic SHARED ${ABC_SOURCE})

target_include_directories(zeno_Alembic PRIVATE .)

find_package(Alembic REQUIRED)
target_link_libraries(zeno_Alembic PRIVATE Alembic::Alembic)
target_link_libraries(zeno_Alembic PRIVATE zeno)
