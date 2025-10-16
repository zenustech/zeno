#include "zpyobject.h"
#include "apiutil.h"
#include <zeno/core/Graph.h>
#include <zeno/types/UserData.h>
#include <zeno/utils/interfaceutil.h>
#include "zeno_types/reflect/reflection.generated.hpp"


#define THROW_WHEN_CORE_DESTROYED(wpObj) \
auto spNode = wpObj;\
if (!spNode) {\
    throw std::runtime_error("the node has been destroyed in core data");\
}

Zpy_Object::Zpy_Object(zeno::zany&& obj) : m_wpObject(std::move(obj)) {

}

VAR_USER_DATA Zpy_Object::get_user_data(const std::string& key) {
    auto spObject = m_wpObject.get();
    if (!spObject) {
        throw std::runtime_error("object has been destroyed");
    }
    zeno::UserData* pUserData = static_cast<zeno::UserData*>(spObject->m_usrData.get());
    auto iter = pUserData->m_data.find(key);
    if (iter == pUserData->m_data.end())
        throw std::runtime_error("the key \"" + key + "\" doesn't exist");

    const zeno::zany& dat = iter->second;
    if (auto numobj = dynamic_cast<zeno::NumericObject*>(dat.get())) {
        return std::visit([&](auto&& arg)->VAR_USER_DATA {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, int>) {
                return arg;
            }
            else if constexpr (std::is_same_v<T, float>) {
                return arg;
            }
            else if constexpr (std::is_same_v<T, zeno::vec2i>) {
                return arg;
            }
            else if constexpr (std::is_same_v<T, zeno::vec2f>) {
                return arg;
            }
            else if constexpr (std::is_same_v<T, zeno::vec3f>) {
                return arg;
            }
            else if constexpr (std::is_same_v<T, zeno::vec3i>) {
                return arg;
            }
            else if constexpr (std::is_same_v<T, zeno::vec4i>) {
                return arg;
            }
            else if constexpr (std::is_same_v<T, zeno::vec4f>) {
                return arg;
            }
            }, numobj->value);
    }
    else if (auto strobj = dynamic_cast<zeno::StringObject*>(dat.get())) {
        return strobj->get();
    }
    else {
        throw std::runtime_error("no numeric or string on userdata");
    }
}

void Zpy_Object::set_user_data(const std::string& key, const VAR_USER_DATA& dat) {
    auto spObject = m_wpObject.get();
    if (!spObject) {
        throw std::runtime_error("object has been destroyed");
    }

    zeno::UserData* pUserData = static_cast<zeno::UserData*>(spObject->m_usrData.get());

    std::visit([&](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, int>) {
            pUserData->set2(key, arg);
        }
        else if constexpr (std::is_same_v<T, float>) {
            pUserData->set2(key, arg);
        }
        else if constexpr (std::is_same_v<T, std::string>) {
            pUserData->set2(key, arg);
        }
        else if constexpr (std::is_same_v<T, zeno::vec2i>) {
            pUserData->set2(key, arg);
        }
        else if constexpr (std::is_same_v<T, zeno::vec2f>) {
            pUserData->set2(key, arg);
        }
        else if constexpr (std::is_same_v<T, zeno::vec3f>) {
            pUserData->set2(key, arg);
        }
        else if constexpr (std::is_same_v<T, zeno::vec3i>) {
            pUserData->set2(key, arg);
        }
        else if constexpr (std::is_same_v<T, zeno::vec4i>) {
            pUserData->set2(key, arg);
        }
        else if constexpr (std::is_same_v<T, zeno::vec4f>) {
            pUserData->set2(key, arg);
        }
        }, dat);
}

std::vector<Zpy_Object> Zpy_Object::toList() const {
    auto spObject = m_wpObject.get();
    if (!spObject) {
        throw std::runtime_error("object has been destroyed");
    }
    if (auto spList = dynamic_cast<zeno::ListObject*>(spObject)) {
        std::vector<Zpy_Object> vec;
        for (auto spObj : spList->get()) {
            vec.push_back(Zpy_Object(spObj->clone()));
        }
        return vec;
    }
    else {
        throw std::runtime_error("the object is not a list object");
    }
}


Zpy_Camera::Zpy_Camera(
    py::list pos,
    py::list up,
    py::list view,
    float fov,
    float aperture,
    float focalPlaneDistance
)
{
    //先默认在mainGraph里创建，避免还要指定一个graph这种麻烦（而且Camera这种大概率只要在main创建）
    auto spNode = zeno::getSession().mainGraph->createNode("MakeCamera");
    if (!spNode) {
        throw std::runtime_error("cannot create camera because of internal error");
    }
    m_wpNode = spNode;

    zeno::vecvar _pos = zpyapi::pylist2vec(pos);
    if (_pos.size() != 3) { throw std::runtime_error("error dims of `pos`,which should be 3."); }

    zeno::vecvar _up = zpyapi::pylist2vec(up);
    if (_up.size() != 3) { throw std::runtime_error("error dims of `up`,which should be 3."); }

    zeno::vecvar _view = zpyapi::pylist2vec(view);
    if (_view.size() != 3) { throw std::runtime_error("error dims of `view`,which should be 3."); }

    //其他参数暂时不考虑公式的情况
    spNode->update_param("pos", _pos);
    spNode->update_param("up", _up);
    spNode->update_param("view", _view);
    spNode->update_param("fov", fov);
    spNode->update_param("aperture", aperture);
    spNode->update_param("focalPlaneDistance", focalPlaneDistance);
    spNode->set_view(true);

    auto nodename = spNode->get_name();
    zeno::render_reload_info render_;
    zeno::getSession().mainGraph->applyNodes({ nodename }, render_);
}

Zpy_Camera::Zpy_Camera(zeno::NodeImpl* wpNode)
    : m_wpNode(wpNode)
{
}

void Zpy_Camera::run() {
    THROW_WHEN_CORE_DESTROYED(m_wpNode)
    auto nodename = spNode->get_name();
    zeno::render_reload_info render_;
    zeno::getSession().mainGraph->applyNodes({ nodename }, render_);
}

std::unique_ptr<zeno::CameraObject> Zpy_Camera::getCamera() const {
    THROW_WHEN_CORE_DESTROYED(m_wpNode)
    zeno::zany spResObj = spNode->get_default_output_object()->clone();
    return zeno::safe_uniqueptr_cast<zeno::CameraObject>(std::move(spResObj));
}

py::list Zpy_Camera::getPos() const {
    THROW_WHEN_CORE_DESTROYED(m_wpNode)
    auto camera = getCamera();
    if (!camera) throw std::runtime_error("the camera object cannot be created.");
    return zpyapi::vec2pylist(camera->pos);
}

void Zpy_Camera::setPos(py::list v) {
    THROW_WHEN_CORE_DESTROYED(m_wpNode)
    spNode->update_param("pos", zpyapi::pylist2vec(v));
    run();
}

py::list Zpy_Camera::getUp() const {
    THROW_WHEN_CORE_DESTROYED(m_wpNode)
    auto camera = getCamera();
    if (!camera) throw std::runtime_error("the camera object cannot be created.");
    return zpyapi::vec2pylist(camera->up);
}

void Zpy_Camera::setUp(py::list v) {
    THROW_WHEN_CORE_DESTROYED(m_wpNode)
    spNode->update_param("up", zpyapi::pylist2vec(v));
    run();
}

py::list Zpy_Camera::getView() const {
    THROW_WHEN_CORE_DESTROYED(m_wpNode)
    auto camera = getCamera();
    if (!camera) throw std::runtime_error("the camera object cannot be created.");
    return zpyapi::vec2pylist(camera->view);
}

void Zpy_Camera::setView(py::list v) {
    THROW_WHEN_CORE_DESTROYED(m_wpNode)
    spNode->update_param("view", zpyapi::pylist2vec(v));
    run();
}

float Zpy_Camera::getNear() const {
    THROW_WHEN_CORE_DESTROYED(m_wpNode)
    auto camera = getCamera();
    if (!camera) throw std::runtime_error("the camera object cannot be created.");
    return camera->fnear;
}

void Zpy_Camera::setNear(float near) {
    THROW_WHEN_CORE_DESTROYED(m_wpNode)
    spNode->update_param("near", near);
    run();
}

float Zpy_Camera::getFar() const {
    THROW_WHEN_CORE_DESTROYED(m_wpNode)
    auto camera = getCamera();
    if (!camera) throw std::runtime_error("the camera object cannot be created.");
    return camera->ffar;
}

void Zpy_Camera::setFar(float far) {
    THROW_WHEN_CORE_DESTROYED(m_wpNode)
    spNode->update_param("far", far);
    run();
}

float Zpy_Camera::getFov() const {
    THROW_WHEN_CORE_DESTROYED(m_wpNode)
    auto camera = getCamera();
    if (!camera) throw std::runtime_error("the camera object cannot be created.");
    return camera->fov;
}

void Zpy_Camera::setFov(float fov) {
    THROW_WHEN_CORE_DESTROYED(m_wpNode)
    spNode->update_param("fov", fov);
    run();
}

float Zpy_Camera::getAperture() const {
    THROW_WHEN_CORE_DESTROYED(m_wpNode)
    auto camera = getCamera();
    if (!camera) throw std::runtime_error("the camera object cannot be created.");
    return camera->aperture;
}

void Zpy_Camera::setAperture(float aperture) {
    THROW_WHEN_CORE_DESTROYED(m_wpNode)
    spNode->update_param("aperture", aperture);
    run();
}

float Zpy_Camera::getFocalPlaneDistance() const {
    THROW_WHEN_CORE_DESTROYED(m_wpNode)
    auto camera = getCamera();
    if (!camera) throw std::runtime_error("the camera object cannot be created.");
    return camera->focalPlaneDistance;
}

void Zpy_Camera::setFocalPlaneDistance(float focal) {
    THROW_WHEN_CORE_DESTROYED(m_wpNode)
    spNode->update_param("focalPlaneDistance", focal);
    run();
}


Zpy_Light::Zpy_Light(
    py::list pos,
    py::list scale,
    py::list rotate,
    py::list color,
    float intensity
)
{
    //先默认在mainGraph里创建，避免还要指定一个graph这种麻烦（而且Camera这种大概率只要在main创建）
    auto spNode = zeno::getSession().mainGraph->createNode("LightNode");
    if (!spNode) {
        throw std::runtime_error("cannot create light because of internal error");
    }
    m_wpNode = spNode;

    zeno::vecvar _pos = zpyapi::pylist2vec(pos);
    if (_pos.size() != 3) { throw std::runtime_error("error dims of `pos`, which should be 3."); }

    zeno::vecvar _scale = zpyapi::pylist2vec(scale);
    if (_scale.size() != 3) { throw std::runtime_error("error dims of `scale`, which should be 3."); }

    zeno::vecvar _rotate = zpyapi::pylist2vec(rotate);
    if (_rotate.size() != 3) { throw std::runtime_error("error dims of `rotate`, which should be 3."); }

    zeno::vecvar _color = zpyapi::pylist2vec(color);
    if (_color.size() != 3) { throw std::runtime_error("error dims of `color`, which should be 3."); }

    //其他参数暂时不考虑公式的情况
    spNode->update_param("position", _pos);
    spNode->update_param("scale", _scale);
    spNode->update_param("rotate", _rotate);
    spNode->update_param("color", _color);
    spNode->update_param("intensity", intensity);
    spNode->set_view(true);

    auto nodename = spNode->get_name();
    zeno::render_reload_info render_;
    zeno::getSession().mainGraph->applyNodes({ nodename }, render_);
}

Zpy_Light::Zpy_Light(zeno::NodeImpl* wpNode)
    : m_wpNode(wpNode)
{
}

py::list Zpy_Light::getPos() const {
    auto light = getLight();
    if (!light) throw std::runtime_error("the light object cannot be created.");
    auto ud = light->userData();
    zeno::vec3f v = zeno::toVec3f(ud->get_vec3f("pos"));
    return zpyapi::vec2pylist(v);
}

void Zpy_Light::setPos(py::list v) {
    THROW_WHEN_CORE_DESTROYED(m_wpNode)
    spNode->update_param("position", zpyapi::pylist2vec(v));
    run();
}

py::list Zpy_Light::getScale() const {
    auto light = getLight();
    if (!light) throw std::runtime_error("the light object cannot be created.");
    auto ud = light->userData();
    zeno::vec3f v = zeno::toVec3f(ud->get_vec3f("scale"));
    return zpyapi::vec2pylist(v);
}

void Zpy_Light::setScale(py::list v) {
    THROW_WHEN_CORE_DESTROYED(m_wpNode)
    spNode->update_param("scale", zpyapi::pylist2vec(v));
    run();
}

py::list Zpy_Light::getRotate() const {
    auto light = getLight();
    if (!light) throw std::runtime_error("the light object cannot be created.");
    auto ud = light->userData();
    zeno::vec3f v = zeno::toVec3f(ud->get_vec3f("rotate"));
    return zpyapi::vec2pylist(v);
}

void Zpy_Light::setRotate(py::list v) {
    THROW_WHEN_CORE_DESTROYED(m_wpNode)
    spNode->update_param("rotate", zpyapi::pylist2vec(v));
    run();
}

py::list Zpy_Light::getColor() const {
    auto light = getLight();
    if (!light) throw std::runtime_error("the light object cannot be created.");
    auto ud = light->userData();
    zeno::vec3f v = zeno::toVec3f(ud->get_vec3f("color"));
    return zpyapi::vec2pylist(v);
}

void Zpy_Light::setColor(py::list v) {
    THROW_WHEN_CORE_DESTROYED(m_wpNode)
    spNode->update_param("color", zpyapi::pylist2vec(v));
    run();
}

float Zpy_Light::getIntensity() const {
    auto light = getLight();
    if (!light) throw std::runtime_error("the light object cannot be created.");
    auto ud = light->userData();
    return ud->get_float("intensity");
}

void Zpy_Light::setIntensity(float intensity) {
    THROW_WHEN_CORE_DESTROYED(m_wpNode)
    spNode->update_param("intensity", intensity);
    run();
}

void Zpy_Light::run() {
    THROW_WHEN_CORE_DESTROYED(m_wpNode)
    auto nodename = spNode->get_name();
    zeno::render_reload_info render_;
    zeno::getSession().mainGraph->applyNodes({ nodename }, render_);
}

std::shared_ptr<zeno::PrimitiveObject> Zpy_Light::getLight() const {
    THROW_WHEN_CORE_DESTROYED(m_wpNode)

    auto pObject = spNode->get_default_output_object();
    if (pObject) {
        zeno::zany spResObj = pObject->clone();
        return zeno::safe_uniqueptr_cast<zeno::PrimitiveObject>(std::move(spResObj));
    }
    return nullptr;
}
