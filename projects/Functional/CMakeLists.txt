


cmake_minimum_required(VERSION 3.10)
add_library(zeno_Functional SHARED 
    NonlinearSolver.cpp
)

add_subdirectory(Algebra)
target_link_libraries(zeno_Functional PUBLIC IBFE)
target_link_libraries(zeno_Functional PRIVATE zeno)
target_link_libraries(zeno_Functional PRIVATE zeno_nodep)

# cmake -B build -DEXTENSION_Functional:BOOL=ON