#include "zpynode.h"
#include <zeno/core/Graph.h>
#include "apiutil.h"
#include <zeno/utils/helper.h>
#include "zeno_types/reflect/reflection.generated.hpp"


using namespace zeno::reflect;

#define THROW_WHEN_CORE_DESTROYED \
auto spNode = m_wpNode;\
if (!spNode) {\
    throw std::runtime_error("the node has been destroyed in core data");\
}


Zpy_Node::Zpy_Node(zeno::NodeImpl* spNode)
    : m_wpNode(spNode)
{
}

void Zpy_Node::set_name(const std::string& name)
{
    THROW_WHEN_CORE_DESTROYED
    auto oldname = spNode->get_name();
    if (auto spGraph = spNode->getGraph()) {
        spGraph->updateNodeName(oldname, name);
    }
}

std::string Zpy_Node::get_name() const
{
    THROW_WHEN_CORE_DESTROYED
    return spNode->get_name();
}

void Zpy_Node::set_view(bool bOn) {
    THROW_WHEN_CORE_DESTROYED
    spNode->set_view(bOn);
}

bool Zpy_Node::is_view() const {
    THROW_WHEN_CORE_DESTROYED
    return spNode->is_view();
}

static py::object primvar2pyobj(const zeno::PrimVar& var) {
    return std::visit([=](auto&& arg)->py::object {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, int> || std::is_same_v<T, float> ||
                std::is_same_v<T, std::string>) {
            return py::cast(arg);
        }
        else if constexpr (std::is_same_v<T, zeno::CurveData>) {
            //TODO
            return py::none();
        }
        else {
            throw makeError<UnimplError>();
        }
    }, var);
}

void Zpy_Node::set_pos(const py::tuple pos) {
    if (pos.size() != 2) {
        throw std::runtime_error("the set_pos only accept two params");
    }
    float xp = pos[0].cast<float>();
    float yp = pos[1].cast<float>();
    THROW_WHEN_CORE_DESTROYED
    spNode->set_pos({xp, yp});
}

py::tuple Zpy_Node::get_pos() const {
    THROW_WHEN_CORE_DESTROYED
    auto pos = spNode->get_pos();
    return py::make_tuple(pos.first, pos.second);
}

py::object Zpy_Node::param_value(const std::string& name) {
    THROW_WHEN_CORE_DESTROYED
    bool bExisted = false;
    zeno::ParamPrimitive param = spNode->get_input_prim_param(name, &bExisted);
    if (!bExisted) {
        throw std::runtime_error("the param not existed");
    }

    zeno::ParamType anyType = param.defl.type().hash_code();
    if (anyType == zeno::types::gParamType_PrimVariant) {
        const auto& primvar = zeno::reflect::any_cast<zeno::PrimVar>(param.defl);
        return primvar2pyobj(primvar);
    }
    else if (anyType == zeno::types::gParamType_VecEdit) {
        const auto& vec = zeno::reflect::any_cast<zeno::vecvar>(param.defl);
        py::list my_list;
        for (auto val : vec) {
            py::object obj = primvar2pyobj(val);
            my_list.append(obj);
        }
        return my_list;
    }
    else if (anyType == zeno::types::gParamType_Int) {
        return py::cast(any_cast<int>(param.defl));
    }
    else if (anyType == zeno::types::gParamType_Float) {
        return py::cast(any_cast<float>(param.defl));
    }
    else if (anyType == zeno::types::gParamType_String) {
        return py::cast(zeno::any_cast_to_string(param.defl));
    }
    else if (anyType == zeno::types::gParamType_Vec2f) {
        zeno::vec2f vec = any_cast<zeno::vec2f>(param.defl);
        return py::list(py::make_tuple(vec[0], vec[1]));
    }
    else if (anyType == zeno::types::gParamType_Vec2i) {
        zeno::vec2i vec = any_cast<zeno::vec2i>(param.defl);
        return py::list(py::make_tuple(vec[0], vec[1]));
    }
    else if (anyType == zeno::types::gParamType_Vec3f) {
        zeno::vec3f vec = any_cast<zeno::vec3f>(param.defl);
        return py::list(py::make_tuple(vec[0], vec[1], vec[2]));
    }
    else if (anyType == zeno::types::gParamType_Vec3i) {
        zeno::vec3i vec = any_cast<zeno::vec3i>(param.defl);
        return py::list(py::make_tuple(vec[0], vec[1], vec[2]));
    }
    else if (anyType == zeno::types::gParamType_Vec4f) {
        zeno::vec4f vec = any_cast<zeno::vec4f>(param.defl);
        return py::list(py::make_tuple(vec[0], vec[1], vec[2], vec[3]));
    }
    else if (anyType == zeno::types::gParamType_Vec4i) {
        zeno::vec4i vec = any_cast<zeno::vec4i>(param.defl);
        return py::list(py::make_tuple(vec[0], vec[1], vec[2], vec[3]));
    }
    else {
        return py::none();
    }
}

void Zpy_Node::update_param(const std::string& name, py::object obj) {
    THROW_WHEN_CORE_DESTROYED
    if (py::isinstance<py::int_>(obj)) {
        int value = obj.cast<int>();
        spNode->update_param(name, value);
        //py::print("Received an integer:", value);
    }
    else if (py::isinstance<py::float_>(obj)) {
        float value = obj.cast<float>();
        spNode->update_param(name, value);
        //py::print("Received a float:", value);
    }
    else if (py::isinstance<py::str>(obj)) {
        std::string value = obj.cast<std::string>();
        spNode->update_param(name, value);
        //py::print("Received a string:", value);
    }
    else if (py::isinstance<py::list>(obj)) {
        const py::list& lst = obj.cast<py::list>();
        zeno::vecvar vec = zpyapi::pylist2vec(lst);
        spNode->update_param(name, vec);
    }
    else {
        py::print("Received an unknown type:", obj);
    }
}
