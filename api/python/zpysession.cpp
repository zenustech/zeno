#ifdef ZENO_PYAPI_STATICLIB
#include <Python.h>
#endif
#include <pybind11/pybind11.h>
#include <string>
#include <zeno/core/Session.h>
#include <zeno/core/Graph.h>
#include "zpyobject.h"
#include "zpygraph.h"
#include <memory>


#if 1
class MyClass {
public:
    MyClass(const std::string& name, int value)
        : name(name)
        , value(value)
        , m_pSess(std::make_unique<zeno::Session>())
    {
        m_pSess = std::make_unique<zeno::Session>();
    }

    void setValue(int v) { value = v; }
    int getValue() const { return value; }

    std::string getName() const { return name; }

    std::string testzeno() {
        return m_pSess->mainGraph->getName();
    }

private:
    std::string name;
    int value;
    std::unique_ptr<zeno::Session> m_pSess;
};
#endif

class Zpy_Session {
public:
    Zpy_Session() : sess(zeno::getSession()) {
    }

    Zpy_Graph mainGraph() const {
        return Zpy_Graph(sess.mainGraph);
    }
    void run() {
        zeno::render_reload_info _;
        sess.run("", _);
    }
private:
    zeno::Session& sess;
};


namespace py = pybind11;

PYBIND11_MODULE(zen, m) {  // `ze` 是python里import的模块名
    m.def("mainGraph", []() -> Zpy_Graph {
        return Zpy_Graph(zeno::getSession().mainGraph);
    });

    py::class_<MyClass>(m, "MyClass")
        .def(py::init<const std::string&, int>())  // 绑定构造函数
        .def("setValue", &MyClass::setValue)       // 绑定成员函数
        .def("getValue", &MyClass::getValue)
        .def("getName", &MyClass::testzeno);

    py::class_<Zpy_Object>(m, "ZObject")
        .def("setUserData", &Zpy_Object::set_user_data)
        .def("getUserData", &Zpy_Object::get_user_data)
        .def("toList", &Zpy_Object::toList);

    py::class_<Zpy_Light>(m, "Light")
        .def(py::init<py::list, py::list, py::list, py::list, float>(),
            py::arg("pos") = py::list(py::make_tuple(0, 0, 0)),
            py::arg("scale") = py::list(py::make_tuple(1, 1, 1)),
            py::arg("rotate") = py::list(py::make_tuple(0, 0, 0)),
            py::arg("color") = py::list(py::make_tuple(1, 1, 1)),
            py::arg("intensity") = 1.0f
        )
        .def("setPos", &Zpy_Light::setPos)
        .def("getPos", &Zpy_Light::getPos)
        .def_property("pos", &Zpy_Light::getPos, &Zpy_Light::setPos)
        /**/
        .def("setScale", &Zpy_Light::setScale)
        .def("getScale", &Zpy_Light::getScale)
        .def_property("scale", &Zpy_Light::getScale, &Zpy_Light::setScale)
        /**/
        .def("setRotate", &Zpy_Light::setRotate)
        .def("getRotate", &Zpy_Light::getRotate)
        .def_property("rotate", &Zpy_Light::getRotate, &Zpy_Light::setRotate)
        /**/
        .def("setColor", &Zpy_Light::setColor)
        .def("getColor", &Zpy_Light::getColor)
        .def_property("color", &Zpy_Light::getColor, &Zpy_Light::setColor)
        /**/
        .def("setIntensity", &Zpy_Light::setIntensity)
        .def("getIntensity", &Zpy_Light::getIntensity)
        .def_property("intensity", &Zpy_Light::getIntensity, &Zpy_Light::setIntensity)
        ;

    py::class_<Zpy_Camera>(m, "Camera")
        .def(py::init<py::list, py::list, py::list, float, float, float>(),
            py::arg("pos") = py::list(py::make_tuple(0, 0, 5)),
            py::arg("up") = py::list(py::make_tuple(0, 1, 0)),
            py::arg("view") = py::list(py::make_tuple(0, 0, -1)),
            py::arg("fov") = 45.f,
            py::arg("aperture") = 11.0f,
            py::arg("focalPlaneDistance") = 2.0f
            )
        .def("setPos", &Zpy_Camera::setPos)
        .def("getPos", &Zpy_Camera::getPos)
        .def_property("pos", &Zpy_Camera::getPos, &Zpy_Camera::setPos)
        /**/
        .def("setUp", &Zpy_Camera::setUp)
        .def("getUp", &Zpy_Camera::getUp)
        .def_property("up", &Zpy_Camera::getUp, &Zpy_Camera::setUp)
        /**/
        .def("setView", &Zpy_Camera::setView)
        .def("getView", &Zpy_Camera::getView)
        .def_property("view", &Zpy_Camera::getView, &Zpy_Camera::setView)
        /**/
        .def("setFar", &Zpy_Camera::setFar)
        .def("getFar", &Zpy_Camera::getFar)
        .def_property("far", &Zpy_Camera::getFar, &Zpy_Camera::setFar)
        /**/
        .def("setFov", &Zpy_Camera::setFov)
        .def("getFov", &Zpy_Camera::getFov)
        .def_property("fov", &Zpy_Camera::getFov, &Zpy_Camera::setFov)
        /**/
        .def("setAperture", &Zpy_Camera::setAperture)
        .def("getAperture", &Zpy_Camera::getAperture)
        .def_property("aperture", &Zpy_Camera::getAperture, &Zpy_Camera::setAperture)
        /**/
        .def("setFocalPlaneDistance", &Zpy_Camera::setFocalPlaneDistance)
        .def("getFocalPlaneDistance", &Zpy_Camera::getFocalPlaneDistance)
        .def_property("focalPlaneDistance", &Zpy_Camera::getFocalPlaneDistance, &Zpy_Camera::setFocalPlaneDistance)
        ;

    py::class_<Zpy_Node>(m, "ZNode")     //TODO: 如果用户想直接构造，那需要指定一张图
        .def("setName", &Zpy_Node::set_name)
        .def("getName", &Zpy_Node::get_name)
        .def_property("name", &Zpy_Node::get_name, &Zpy_Node::set_name)
        .def("setView", &Zpy_Node::set_view)
        .def("isView", &Zpy_Node::is_view)
        .def_property("view", &Zpy_Node::is_view, &Zpy_Node::set_view)
        .def("updateParam", &Zpy_Node::update_param)
        .def("paramValue", &Zpy_Node::param_value)
        .def("setPos", &Zpy_Node::set_pos)
        .def("getPos", &Zpy_Node::get_pos)
        .def_property("pos", &Zpy_Node::get_pos, &Zpy_Node::set_pos)
        ;

    py::class_<Zpy_Graph>(m, "ZGraph")
        .def("createNode", &Zpy_Graph::createNode,
            py::arg("nodecls"),
            py::arg("originalname") = "",
            py::arg("bAssets") = false)
        .def("getName", &Zpy_Graph::getName)
        .def_property_readonly("name", &Zpy_Graph::getName)
        .def("getNode", &Zpy_Graph::getNode)
        .def("getInputObject", &Zpy_Graph::getInputObject)
        .def("removeNode", &Zpy_Graph::removeNode)
        .def("addEdge", &Zpy_Graph::addEdge)
        .def("removeEdge", &Zpy_Graph::removeEdge)
        .def("getCamera", &Zpy_Graph::getCamera)
        .def("getLight", &Zpy_Graph::getLight)
        ;

    py::class_<Zpy_Session>(m, "Session")
        .def(py::init<>())
        .def("mainGraph", &Zpy_Session::mainGraph)
        .def("run", &Zpy_Session::run)
        ;

}
