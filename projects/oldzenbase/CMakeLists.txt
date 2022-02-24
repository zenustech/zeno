file(GLOB PROJECT_SOURCE include/*/*.h *.cpp mesh/*.cpp pars/*.cpp prim/*.cpp *.cpp)

add_library(zeno_oldzenbase OBJECT ${PROJECT_SOURCE})
target_compile_definitions(zeno_oldzenbase PRIVATE -DDLL_ZENBASE)
target_include_directories(zeno_oldzenbase PUBLIC include)

find_package(OpenMP)
if (TARGET OpenMP::OpenMP_CXX)
    message("found package: OpenMP::OpenMP_CXX")
    target_link_libraries(zeno_oldzenbase PRIVATE OpenMP::OpenMP_CXX)
endif()
#find_package(TBB CONFIG REQUIRED COMPONENTS tbb tbbmalloc)
#target_link_libraries(zeno_oldzenbase PRIVATE TBB::tbb TBB::tbbmalloc)

target_link_libraries(zeno_oldzenbase PUBLIC zeno)
target_link_libraries(zeno_oldzenbase PRIVATE zeno_nodep)
