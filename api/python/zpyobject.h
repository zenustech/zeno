#pragma once

#ifndef __ZPYOBJECT_H__
#define __ZPYOBJECT_H__

#ifdef ZENO_PYAPI_STATICLIB
#include <Python.h>
#endif

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <zeno/core/NodeImpl.h>
#include <zeno/types/CameraObject.h>
#include <zeno/types/PrimitiveObject.h>

namespace py = pybind11;

using VAR_USER_DATA = std::variant<int, float, std::string,
    zeno::vec2i, zeno::vec3i, zeno::vec4i,
    zeno::vec2f, zeno::vec3f, zeno::vec4f>;


class Zpy_Object {
public:
    Zpy_Object(zeno::zany&& obj);
    VAR_USER_DATA get_user_data(const std::string& key);
    void set_user_data(const std::string& key, const VAR_USER_DATA& dat);
    std::vector<Zpy_Object> toList() const;

private:
    zeno::zany m_wpObject;
};


class Zpy_Camera {
public:
    Zpy_Camera(
        py::list pos,
        py::list up,
        py::list view,
        float fov,
        float aperture,
        float focalPlaneDistance
    );
    Zpy_Camera(zeno::NodeImpl* wpNode);
    py::list getPos() const;
    void setPos(py::list v);
    py::list getUp() const;
    void setUp(py::list v);
    py::list getView() const;
    void setView(py::list v);
    float getNear() const;
    void setNear(float near);
    float getFar() const;
    void setFar(float far);
    float getFov() const;
    void setFov(float fov);
    float getAperture() const;
    void setAperture(float aperture);
    float getFocalPlaneDistance() const;
    void setFocalPlaneDistance(float focal);
    std::unique_ptr<zeno::CameraObject> getCamera() const;

private:
    void run();

    zeno::NodeImpl* m_wpNode;
};

class Zpy_Light {
public:
    Zpy_Light(
        py::list pos,
        py::list scale,
        py::list rotate,
        py::list color,
        float intensity
    );
    Zpy_Light(zeno::NodeImpl* wpNode);
    py::list getPos() const;
    void setPos(py::list v);
    py::list getScale() const;
    void setScale(py::list v);
    py::list getRotate() const;
    void setRotate(py::list v);
    py::list getColor() const;
    void setColor(py::list v);
    float getIntensity() const;
    void setIntensity(float intensity);     //后续要考虑公式，不过上层python也可以计算公式（除了$F这种）
    std::shared_ptr<zeno::PrimitiveObject> getLight() const;

private:
    void run();

    zeno::NodeImpl* m_wpNode;
};

#endif