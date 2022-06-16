find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)

file(GLOB SRC_LIST *.cpp *.h ./calcUVCore/*.cpp ./calcUVCore/*.h)

target_include_directories(zeno PRIVATE . ./calcUVCore)
target_sources(zeno PRIVATE ${SRC_LIST})
target_link_libraries(zeno PRIVATE Threads::Threads)
target_link_libraries(zeno PRIVATE Eigen3::Eigen)
