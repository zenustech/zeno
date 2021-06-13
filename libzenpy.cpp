#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <zen/zen.h>
namespace py = pybind11;


PYBIND11_MODULE(libzenpy, m) {
  m.def("dumpDescriptors", zen::dumpDescriptors);
  m.def("bindNodeInput", zen::bindNodeInput);
  m.def("setNodeParam", zen::setNodeParam);
  m.def("clearNodes", zen::clearNodes);
  m.def("applyNode", zen::applyNode);
  m.def("addNode", zen::addNode);

  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p)
        std::rethrow_exception(p);
    } catch (zen::Exception const &e) {
      PyErr_SetString(PyExc_RuntimeError, e.what());
    }
  });
}
