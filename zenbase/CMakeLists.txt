file(GLOB PROJECT_SOURCE include/*/*.h *.cpp mesh/*.cpp pars/*.cpp prim/*.cpp)

add_library(zenbase SHARED ${PROJECT_SOURCE})
target_include_directories(zenbase PRIVATE ../include)
target_include_directories(zenbase PUBLIC include)

find_package(OpenMP)
if (TARGET OpenMP::OpenMP_CXX)
    message("found package: OpenMP::OpenMP_CXX")
    target_link_libraries(zenbase PRIVATE OpenMP::OpenMP_CXX)
endif()

target_link_libraries(zenbase PRIVATE zen)
