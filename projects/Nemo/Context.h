/*
 * MIT License
 *
 * Copyright (c) 2024 wuzhen
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * 1. The above copyright notice and this permission notice shall be included in
 *    all copies or substantial portions of the Software.
 *
 * 2. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *    SOFTWARE.
 */

#pragma once

#include <glm/glm.hpp>
#include <string>
#include <tuple>
#include <vector>

#ifdef WIN32
#include <Windows.h>
using MOD = HMODULE;
template <typename T> T get_fn(HMODULE mod, const char *name) { return (T)GetProcAddress(mod, name); }
#else
#include <dlfcn.h>
using MOD = void *;
template <typename T> T get_fn(void *mod, const char *name) { return (T)dlsym(mod, name); }
#endif

namespace nemo {
struct DataStorage {
  void init(MOD mod);

  void cleanup();

  void resize(std::string key, std::size_t size, bool sp) { fnResize(instance, fnTypeid(key), size, sp); }

  unsigned typeidFromStr(std::string key) const { return fnTypeid(key); }

  unsigned globalIdOffset(std::string key) const { return fnGlobalIdOffset(instance, fnTypeid(key)); }

  template <typename... Args> void setCurve(Args &&...args) { fnSetCurve(instance, std::forward<Args>(args)...); }
  template <typename... Args> void getCurve(Args &&...args) const { fnGetCurve(instance, std::forward<Args>(args)...); }
  template <typename... Args> void setDCurve(Args &&...args) { fnSetDCurve(instance, std::forward<Args>(args)...); }
  template <typename... Args> void getDCurve(Args &&...args) { fnGetDCurve(instance, std::forward<Args>(args)...); }
  template <typename... Args> void setSurface(Args &&...args) { fnSetSurface(instance, std::forward<Args>(args)...); }
  template <typename... Args> void getSurface(Args &&...args) const { fnGetSurface(instance, std::forward<Args>(args)...); }
  template <typename... Args> void setDSurface(Args &&...args) { fnSetDSurface(instance, std::forward<Args>(args)...); }
  template <typename... Args> void getDSurface(Args &&...args) { fnGetDSurface(instance, std::forward<Args>(args)...); }
  template <typename... Args> void setMesh(Args &&...args) const { fnSetMesh(instance, std::forward<Args>(args)...); }
  template <typename... Args> void getMesh(Args &&...args) const { fnGetMesh(instance, std::forward<Args>(args)...); }
  template <typename... Args> void setDMesh(Args &&...args) const { fnSetDMesh(instance, std::forward<Args>(args)...); }
  template <typename... Args> void getDMesh(Args &&...args) const { fnGetDMesh(instance, std::forward<Args>(args)...); }
  template <typename... Args> void getCuShape(Args &&...args) const { fnGetCuShape(instance, std::forward<Args>(args)...); }
  template <typename... Args> void pullCuShape(Args &&...args) const { fnPullCuShape(instance, std::forward<Args>(args)...); }
  template <typename... Args> void getDCuShape(Args &&...args) const { fnGetDCuShape(instance, std::forward<Args>(args)...); }
  template <typename... Args> void pullDCuShape(Args &&...args) const { fnPullDCuShape(instance, std::forward<Args>(args)...); }

#define ADD_DATA_READ(K, T)                                                                                                                                    \
  T get##K(unsigned id) const { return fnGet##K(instance, id); }                                                                                               \
  T (*fnGet##K)(void *inst, unsigned idx) = nullptr;

#define ADD_DATA_WRITE(K, T)                                                                                                                                   \
  void set##K(unsigned id, T val) { fnSet##K(instance, id, val); }                                                                                             \
  void (*fnSet##K)(void *inst, unsigned idx, T value) = nullptr;

#define ADD_DATA_TYPE(K, T)                                                                                                                                    \
  ADD_DATA_READ(K, T)                                                                                                                                          \
  ADD_DATA_WRITE(K, T)

  ADD_DATA_TYPE(Bool, bool)
  ADD_DATA_TYPE(Float, float)
  ADD_DATA_TYPE(Double, double)
  ADD_DATA_TYPE(Int, int)
  ADD_DATA_WRITE(Vec2, glm::vec2)
  ADD_DATA_WRITE(DVec2, glm::dvec2)
  ADD_DATA_WRITE(Vec3, glm::vec3)
  ADD_DATA_WRITE(DVec3, glm::dvec3)
  ADD_DATA_WRITE(Mat4, glm::mat4)
  ADD_DATA_WRITE(DMat4, glm::dmat4)

#undef ADD_DATA_TYPE
#undef ADD_DATA_READ
#undef ADD_DATA_WRITE

#define ADD_DATA_READ(K, T)                                                                                                                                    \
  T get##K(unsigned id) const {                                                                                                                                \
    T v;                                                                                                                                                       \
    fnGet##K(instance, id, v);                                                                                                                                 \
    return v;                                                                                                                                                  \
  }                                                                                                                                                            \
  void (*fnGet##K)(void *inst, unsigned idx, T &v) = nullptr;

  ADD_DATA_READ(Vec3, glm::vec3);
  ADD_DATA_READ(DVec3, glm::dvec3);
  ADD_DATA_READ(Mat4, glm::mat4);
  ADD_DATA_READ(DMat4, glm::dmat4);

#undef ADD_DATA_READ

  void *instance = nullptr;

private:
  void *(*fnNew)() = nullptr;
  void (*fnFree)(void *) = nullptr;
  void (*fnResize)(void *inst, unsigned datatype, std::size_t size, bool sp) = nullptr;
  unsigned (*fnGlobalIdOffset)(void *inst, unsigned datatype) = nullptr;
  unsigned (*fnTypeid)(std::string name) = nullptr;

  void (*fnSetCurve)(void *inst, unsigned idx, unsigned degree, unsigned form, std::vector<glm::vec3> cv, std::vector<float> knots, glm::mat4 matrix) = nullptr;
  void (*fnGetCurve)(void *inst, unsigned idx, unsigned &degree, unsigned &form, std::vector<glm::vec3> &cv, std::vector<float> &knots) = nullptr;
  void (*fnSetDCurve)(void *inst, unsigned idx, unsigned degree, unsigned form, std::vector<glm::dvec3> cv, std::vector<double> knots,
                      glm::dmat4 matrix) = nullptr;
  void (*fnGetDCurve)(void *inst, unsigned idx, unsigned &degree, unsigned &form, std::vector<glm::dvec3> &cv, std::vector<double> &knots) = nullptr;

  void (*fnSetSurface)(void *inst, unsigned idx, std::vector<glm::vec3> cv, unsigned degreeU, unsigned formU, std::vector<float> knotsU, unsigned degreeV,
                       unsigned formV, std::vector<float> knotsV, glm::mat4 matrix) = nullptr;
  void (*fnGetSurface)(void *inst, unsigned idx, std::vector<glm::vec3> &cv, unsigned &degreeU, unsigned &formU, std::vector<float> &knotsU, unsigned &degreeV,
                       unsigned &formV, std::vector<float> &knotsV) = nullptr;
  void (*fnSetDSurface)(void *inst, unsigned idx, std::vector<glm::dvec3> cv, unsigned degreeU, unsigned formU, std::vector<double> knotsU, unsigned degreeV,
                        unsigned formV, std::vector<double> knotsV, glm::dmat4 matrix) = nullptr;
  void (*fnGetDSurface)(void *inst, unsigned idx, std::vector<glm::dvec3> &cv, unsigned &degreeU, unsigned &formU, std::vector<double> &knotsU,
                        unsigned &degreeV, unsigned &formV, std::vector<double> &knotsV) = nullptr;

  void (*fnSetMesh)(void *inst, unsigned idx, std::vector<glm::vec3> points, glm::mat4 matrix) = nullptr;
  void (*fnGetMesh)(void *inst, unsigned idx, std::vector<glm::vec3> &points) = nullptr;
  void (*fnSetDMesh)(void *inst, unsigned idx, std::vector<glm::dvec3> points, glm::mat4 matrix) = nullptr;
  void (*fnGetDMesh)(void *inst, unsigned idx, std::vector<glm::dvec3> &points) = nullptr;
  void (*fnGetCuShape)(void *inst, unsigned idx, glm::vec3 *&data, unsigned &size) = nullptr;
  void (*fnPullCuShape)(void *inst, unsigned idx, std::vector<glm::vec3> &points) = nullptr;
  void (*fnGetDCuShape)(void *inst, unsigned idx, glm::vec3 *&data, unsigned &size) = nullptr;
  void (*fnPullDCuShape)(void *inst, unsigned idx, std::vector<glm::vec3> &points) = nullptr;
};

struct ResourcePool {
  void cleanup();

  void init(MOD mod, std::string path);

  std::pair<std::vector<unsigned>, std::vector<unsigned>> getTopo(unsigned id) {
    std::vector<unsigned> counts, connection;
    fnGetTopo(instance, id, counts, connection);
    return {counts, connection};
  }

  std::tuple<std::vector<float>, std::vector<float>, std::vector<unsigned>> getUV(unsigned id) {
    std::vector<float> uValues, vValues;
    std::vector<unsigned> uvIds;
    fnGetUV(instance, id, uValues, vValues, uvIds);
    return std::make_tuple(uValues, vValues, uvIds);
  }

  std::pair<std::vector<glm::vec4>, std::vector<unsigned>> getColor(unsigned id) {
    std::vector<glm::vec4> colors;
    std::vector<unsigned> colorIds;
    fnGetColor(instance, id, colors, colorIds);
    return std::make_pair(colors, colorIds);
  }

  std::tuple<std::vector<glm::vec3>, std::vector<unsigned>, std::vector<unsigned>> getUserNormal(unsigned id) {
    std::vector<glm::vec3> normals;
    std::vector<unsigned> faces, vertices;
    fnGetNormal(instance, id, normals, faces, vertices);
    return std::make_tuple(normals, faces, vertices);
  }

  std::vector<unsigned> getUVector(unsigned id) {
    std::vector<unsigned> data;
    fnGetUVector(instance, id, data);
    return data;
  }

  void *instance = nullptr;

private:
  void *(*fnNew)() = nullptr;
  void (*fnFree)(void *res) = nullptr;
  void *(*fnLoad)(void *res, std::string path) = nullptr;
  void (*fnGetTopo)(void *res, unsigned id, std::vector<unsigned> &, std::vector<unsigned> &) = nullptr;
  void (*fnGetUV)(void *res, unsigned id, std::vector<float> &, std::vector<float> &, std::vector<unsigned> &) = nullptr;
  void (*fnGetColor)(void *res, unsigned id, std::vector<glm::vec4> &, std::vector<unsigned> &) = nullptr;
  void (*fnGetNormal)(void *res, unsigned id, std::vector<glm::vec3> &, std::vector<unsigned> &, std::vector<unsigned> &) = nullptr;
  void (*fnGetUVector)(void *res, unsigned id, std::vector<unsigned> &) = nullptr;
};
} // namespace nemo
