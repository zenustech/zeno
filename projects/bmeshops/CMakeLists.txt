file(GLOB PROJECT_SOURCE include/*/*.h *.cpp mesh/*.cpp pars/*.cpp prim/*.cpp *.cpp)

add_library(zeno_bmeshops SHARED ${PROJECT_SOURCE})
target_include_directories(zeno_bmeshops PRIVATE .)

find_package(OpenMP)
if (TARGET OpenMP::OpenMP_CXX)
    message("found package: OpenMP::OpenMP_CXX")
    target_link_libraries(zeno_bmeshops PRIVATE OpenMP::OpenMP_CXX)
endif()

target_link_libraries(zeno_bmeshops PUBLIC zeno)
