file(GLOB PROJECT_SOURCE include/*/*.h *.cpp mesh/*.cpp pars/*.cpp prim/*.cpp)

add_library(zenbase SHARED ${PROJECT_SOURCE})
target_include_directories(zenbase PUBLIC include)

find_package(OpenMP)
if (TARGET OpenMP::OpenMP_CXX)
    message("found package: OpenMP::OpenMP_CXX")
    target_link_libraries(zenbase PRIVATE OpenMP::OpenMP_CXX)
endif()
find_package(TBB CONFIG REQUIRED COMPONENTS tbb tbbmalloc)
target_link_libraries(zenbase PRIVATE TBB::tbb TBB::tbbmalloc)

target_link_libraries(zenbase PRIVATE zen)
set_property(TARGET zenbase PROPERTY LIBRARY_OUTPUT_DIRECTORY ${zen_INSTALL_DIR}/autoload)
set_property(TARGET zenbase PROPERTY RUNTIME_OUTPUT_DIRECTORY ${zen_INSTALL_DIR}/autoload)
