file(GLOB PYZENO_SOURCE *.cpp)
pybind11_add_module(zeno_pybind11_module ${PYZENO_SOURCE})
target_link_libraries(zeno_pybind11_module PRIVATE zeno)
