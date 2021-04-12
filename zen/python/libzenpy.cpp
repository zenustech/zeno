#include <zen/zen.h>
#include <zen/NumpyObject.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
namespace py = pybind11;



static void setNumpyObject(std::string name, py::buffer_info const &buf) {
  auto obj = zen::IObject::make<zenbase::NumpyObject>();
  obj->ptr = buf.ptr;
  obj->itemsize = buf.itemsize;
  obj->format = buf.format;
  obj->ndim = buf.ndim;
  obj->shape = buf.shape;
  obj->strides = buf.strides;

  zen::setObject(name, std::move(obj));
}


template <class T>
void setNumpyObject(std::string name, py::array_t<T> const &data) {
  py::buffer_info buf = data.request();
  setNumpyObject(name, buf);
}


PYBIND11_MODULE(libzenpy, m) {
  m.def("addNode", zen::addNode);
  m.def("setNodeParam", zen::setNodeParam);
  m.def("setNodeInput", zen::setNodeInput);
  m.def("applyNode", zen::applyNode);
  m.def("dumpDescriptors", zen::dumpDescriptors);
#define _DEF_TYPE(x) \
  m.def("setNumpyObject", setNumpyObject<x>);
  _DEF_TYPE(uint8_t)
  _DEF_TYPE(uint16_t)
  _DEF_TYPE(uint32_t)
  _DEF_TYPE(uint64_t)
  _DEF_TYPE(int8_t)
  _DEF_TYPE(int16_t)
  _DEF_TYPE(int32_t)
  _DEF_TYPE(int64_t)
  _DEF_TYPE(float)
  _DEF_TYPE(double)
#undef _DEF_TYPE

  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p)
        std::rethrow_exception(p);
    } catch (zen::Exception const &e) {
      PyErr_SetString(PyExc_RuntimeError, e.what());
    }
  });
}
