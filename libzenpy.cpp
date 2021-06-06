#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <zen/ArrayObject.h>
#include <zen/NumericObject.h>
#include <zen/StringObject.h>
#include <zen/zen.h>
namespace py = pybind11;

std::string getCppObjectType(std::string name) {
  auto obj = zen::getObject(name);
  if (obj->as<zenbase::ArrayObject>()) {
    return "array";
  } else if (obj->as<zen::BooleanObject>()) {
    return "boolean";
  } else if (obj->as<zenbase::StringObject>()) {
    return "string";
  } else if (obj->as<zenbase::NumericObject>()) {
    return "numeric";
  } else {
    return "other";
  }
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

void setNumericObject(std::string name, zenbase::FixedNumericValue value) {
  auto obj = std::make_unique<zenbase::NumericObject>();
  std::visit([&](auto const &x) { obj->value = (decltype(obj->value))x; },
             value);
  zen::setObject(name, std::move(obj));
}

zenbase::FixedNumericValue getNumericObject(std::string name) {
  auto obj = zen::getObject(name)->as<zenbase::NumericObject>();
  zenbase::FixedNumericValue value;
  std::visit([&](auto const &x) { value = (decltype(value))x; }, obj->value);
  return value;
}

void setStringObject(std::string name, std::string value) {
  auto obj = std::make_unique<zenbase::StringObject>();
  obj->value = value;
  zen::setObject(name, std::move(obj));
}

std::string getStringObject(std::string name) {
  auto obj = zen::getObject(name)->as<zenbase::StringObject>();
  return obj->value;
}

static std::map<std::string, std::unique_ptr<std::vector<char>>> savedNumpys;

template <class T>
void setArrayObject(std::string name,
                    py::array_t<T, py::array::c_style> const &data) {
  py::buffer_info buf = data.request();

  auto obj = zen::IObject::make<zenbase::ArrayObject>();
  obj->ptr = buf.ptr;
  obj->itemsize = buf.itemsize;
  obj->format = buf.format;
  obj->ndim = buf.ndim;
  obj->shape = buf.shape;
  obj->strides = buf.strides;

  // make a deep-copy to prevent pointer out-of-date / dirty data:
  ssize_t size = 1;
  for (ssize_t i = 0; i < obj->ndim; i++)
    size += obj->shape[i];
  size *= obj->itemsize;
  auto saved = std::make_unique<std::vector<char>>(size);
  std::memcpy(saved->data(), obj->ptr, size);
  obj->ptr = saved->data();
  savedNumpys[name] = std::move(saved);

  zen::setObject(name, std::move(obj));
  // by the way, will we provide zen::deleteObject as well?
}

std::tuple<uintptr_t, ssize_t, std::string, ssize_t, std::vector<ssize_t>,
           std::vector<ssize_t>>
getArrayObjectMeta(std::string name) {
  auto obj = zen::getObject(name)->as<zenbase::ArrayObject>();
  return std::tuple<uintptr_t, ssize_t, std::string, ssize_t,
                    std::vector<ssize_t>, std::vector<ssize_t>>(
      (uintptr_t)obj->ptr, (ssize_t)obj->itemsize, (std::string)obj->format,
      (ssize_t)obj->ndim, (std::vector<ssize_t>)obj->shape,
      (std::vector<ssize_t>)obj->strides);
};

template <class T>
py::array_t<T, py::array::c_style> getArrayObject(std::string name) {
  auto obj = zen::getObject(name)->as<zenbase::ArrayObject>();

  ssize_t size = 1;
  for (ssize_t i = 0; i < obj->ndim; i++)
    size *= obj->shape[i];
  py::array_t<T, py::array::c_style> arr(size);
  auto acc = arr.mutable_unchecked();
  for (ssize_t i = 0; i < size; i++) {
    acc(i) = *(T *)((uint8_t *)obj->ptr + obj->itemsize * i);
  }
  arr.resize(obj->shape);
  return arr;
}

PYBIND11_MODULE(libzenpy, m) {
  m.def("addNode", zen::addNode);
  m.def("setNodeParam", zen::setNodeParam);
  m.def("setNodeInput", zen::setNodeInput);
  m.def("initNode", zen::initNode);
  m.def("applyNode", zen::applyNode);
  m.def("dumpDescriptors", zen::dumpDescriptors);

  m.def("isCppObject", zen::hasObject);
  m.def("getNodeRequirements", zen::getNodeRequirements);
  m.def("getCppObjectType", getCppObjectType);

  m.def("setBooleanObject", setBooleanObject);
  m.def("getBooleanObject", getBooleanObject);
  m.def("setStringObject", setStringObject);
  m.def("getStringObject", getStringObject);
  m.def("setNumericObject", setNumericObject);
  m.def("getNumericObject", getNumericObject);
  m.def("setReference", zen::setReference);
  m.def("getReference", zen::getReference);

  m.def("getArrayObjectMeta", getArrayObjectMeta);
#define _DEF_TYPE(T)                                                           \
  m.def("setArrayObject", setArrayObject<T>);                                  \
  m.def("getArrayObject_" #T, getArrayObject<T>);
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
