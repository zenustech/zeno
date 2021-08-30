file(GLOB PROJECT_SOURCE *.cpp *.h)

add_library(zeno_blenderio SHARED ${PROJECT_SOURCE})
target_include_directories(zeno_blenderio PRIVATE .)

find_package(OpenMP)
if (TARGET OpenMP::OpenMP_CXX)
    message("found package: OpenMP::OpenMP_CXX")
    target_link_libraries(zeno_blenderio PRIVATE OpenMP::OpenMP_CXX)
endif()

target_link_libraries(zeno_blenderio PUBLIC zeno)
target_link_libraries(zeno_blenderio PRIVATE zeno_nodep)
