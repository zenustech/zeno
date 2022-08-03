find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)

file(GLOB SRC_LIST *.cpp *.h)
file(GLOB CORE_SRC_LIST ./calcUVCore/*.cpp ./calcUVCore/*.h)

add_library(xatlasUVCore STATIC ${CORE_SRC_LIST})

target_include_directories(zeno PRIVATE . ./calcUVCore)
target_sources(zeno PRIVATE ${SRC_LIST})
# target_link_libraries(zeno PRIVATE ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(zeno PRIVATE Threads::Threads)
target_link_libraries(zeno PRIVATE Eigen3::Eigen)
target_link_libraries(zeno PRIVATE xatlasUVCore)
