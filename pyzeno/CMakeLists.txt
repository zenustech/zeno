file(GLOB PYZENO_SOURCE *.cpp)
pybind11_add_module(pylib_zeno ${PYZENO_SOURCE})
target_link_libraries(pylib_zeno PRIVATE zeno)
