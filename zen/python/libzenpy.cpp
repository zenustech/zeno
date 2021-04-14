#include <zen/zen.h>
#include <zen/NumpyObject.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
namespace py = pybind11;


std::string getCppObjectType(std::string name) {
  auto obj = zen::getObject(name);
  if (obj->as<zenbase::NumpyObject>()) {
    return "numpy";
  }
  if (obj->as<zen::BooleanObject>()) {
    return "boolean";
  }
  return "other";
}


void setBooleanObject(std::string name, bool value) {
  auto obj = std::make_unique<zen::BooleanObject>();
  obj->value = value;
  zen::setObject(name, std::move(obj));
}

bool getBooleanObject(std::string name) {
  auto obj = zen::getObject(name)->as<zen::BooleanObject>();
  return obj->value;
}


static std::map<std::string, std::unique_ptr<std::vector<char>>> savedNumpys;

template <class T>
void setNumpyObject(std::string name,
    py::array_t<T, py::array::c_style> const &data) {
  py::buffer_info buf = data.request();

  auto obj = zen::IObject::make<zenbase::NumpyObject>();
  obj->ptr = buf.ptr;
  obj->itemsize = buf.itemsize;
  obj->format = buf.format;
  obj->ndim = buf.ndim;
  obj->shape = buf.shape;
  obj->strides = buf.strides;

  // make a deep-copy to prevent pointer out-of-date / dirty data:
  size_t size = 1;
  for (size_t i = 0; i < obj->ndim; i++)
    size += obj->shape[i];
  size *= obj->itemsize;
  auto saved = std::make_unique<std::vector<char>>(size);
  std::memcpy(saved->data(), obj->ptr, size);
  obj->ptr = saved->data();
  savedNumpys[name] = std::move(saved);

  zen::setObject(name, std::move(obj));
  // by the way, will we provide zen::deleteObject as well?
}

std::tuple<uintptr_t, ssize_t, std::string, ssize_t,
  std::vector<ssize_t>, std::vector<ssize_t>>
  getNumpyObjectMeta(std::string name) {
  auto obj = zen::getObject(name)->as<zenbase::NumpyObject>();
  return std::make_tuple(
      (uintptr_t)obj->ptr, obj->itemsize, obj->format,
      obj->ndim, obj->shape, obj->strides);
};

template <class T>
py::array_t<T, py::array::c_style> getNumpyObject(std::string name) {
  auto obj = zen::getObject(name)->as<zenbase::NumpyObject>();

  size_t size = 1;
  for (size_t i = 0; i < obj->ndim; i++)
    size *= obj->shape[i];
  py::array_t<T, py::array::c_style> arr(size);
  auto acc = arr.mutable_unchecked();
  for (size_t i = 0; i < size; i++) {
    acc(i) = *(T *)((uint8_t *)obj->ptr + obj->itemsize * i);
  }
  arr.resize(obj->shape);
  return arr;
}


PYBIND11_MODULE(libzenpy, m) {
  m.def("addNode", zen::addNode);
  m.def("setNodeParam", zen::setNodeParam);
  m.def("setNodeInput", zen::setNodeInput);
  m.def("applyNode", zen::applyNode);
  m.def("dumpDescriptors", zen::dumpDescriptors);

  m.def("getCppObjectType", getCppObjectType);

  m.def("setBooleanObject", setBooleanObject);
  m.def("getBooleanObject", getBooleanObject);
  m.def("setReference", zen::setReference);
  m.def("getReference", zen::getReference);

  m.def("getNumpyObjectMeta", getNumpyObjectMeta);
#define _DEF_TYPE(T) \
  m.def("setNumpyObject", setNumpyObject<T>); \
  m.def("getNumpyObject_" #T, getNumpyObject<T>);
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
