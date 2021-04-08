#include <zen/zen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

PYBIND11_MODULE(libzenpy, m) {
    m.def("addNode", zen::addNode);
    m.def("setNodeParam", zen::setNodeParam);
    m.def("setNodeInput", zen::setNodeInput);
    m.def("applyNode", zen::applyNode);
    m.def("dumpDescriptors", zen::dumpDescriptors);
}
