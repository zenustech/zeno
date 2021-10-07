#include "zeno.h"
#include <zeno/types/NumericObject.h>
#include <zeno/types/StringObject.h>
#include <z2/dop/execute.h>


namespace zeno {


IObject::IObject() = default;
IObject::~IObject() = default;
std::shared_ptr<IObject> IObject::clone() const {
    return nullptr;
}
std::shared_ptr<IObject> IObject::move_clone() {
    return nullptr;
}
bool IObject::assign(IObject *other) {
    return false;
}
bool IObject::move_assign(IObject *other) {
    return false;
}


bool INode::has_input2(std::string const &name) const {
    for (int i = 0; i < desc->inputs.size(); i++) {
        if (desc->inputs[i].name == name) {
            return Node::get_input(i).has_value();
        }
    }
    return {};
}


void INode::set_output2(std::string const &id, zany &&obj) {
    if (auto num = silent_any_cast<std::shared_ptr<NumericObject>>(obj); num.has_value()) {
        std::visit([&] (auto const &x) {
            _set_output2(id, x);
        }, num.value()->value);
    } else {
        _set_output2(id, std::move(obj));
    }
}


std::shared_ptr<IObject> INode::get_input(std::string const &id, std::string const &msg) const {
    auto obj = get_input2(id);
    if (silent_any_cast<std::shared_ptr<IObject>>(obj).has_value())
        return safe_any_cast<std::shared_ptr<IObject>>(obj, "input `" + id + "` ");

    auto str = std::make_shared<StringObject>();
    if (auto o = exact_any_cast<std::string>(obj); o.has_value()) {
        str->set(o.value());
        return str;
    }

    auto num = std::make_shared<NumericObject>();
    using Types = typename is_variant<NumericValue>::tuple_type;
    if (static_for<0, std::tuple_size_v<Types>>([&] (auto i) {
        using T = std::tuple_element_t<i, Types>;
        if (auto o = exact_any_cast<T>(obj); o.has_value()) {
            num->set(o.value());
            return true;
        }
        return false;
    })) {
        return num;
    } else if (auto o = exact_any_cast<bool>(obj); o.has_value()) {
        num->set((int)o.value());
        return num;
    }

    throw zeno::Exception("expecting `" + msg + "` (IObject ptr) for input `"
            + id + "`, got `" + obj.type().name() + "` (any) [numeric cast also failed]");
}


bool INode::has_input(std::string const &id) const {
    if (!has_input2(id)) return false;
    //return inputBounds.find(id) != inputBounds.end();
    auto obj = get_input2(id);
    if (silent_any_cast<std::shared_ptr<IObject>>(obj).has_value())
        return true;

    if (exact_any_cast<std::string>(obj))
        return true;

    using Types = typename is_variant<NumericValue>::tuple_type;
    if (static_for<0, std::tuple_size_v<Types>>([&] (auto i) {
        using T = std::tuple_element_t<i, Types>;
        if (auto o = exact_any_cast<T>(obj); o.has_value()) {
            return true;
        }
        return false;
    })) {
        return true;
    } else if (auto o = exact_any_cast<bool>(obj); o.has_value()) {
        return true;
    }

    return false;
}


Any INode::get_input2(std::string const &name) const {
    for (int i = 0; i < desc->inputs.size(); i++) {
        if (desc->inputs[i].name == name) {
            return Node::get_input(i);
        }
    }
    return {};
}


void INode::_set_output2(std::string const &name, Any &&val) {
    for (int i = 0; i < desc->inputs.size(); i++) {
        if (desc->outputs[i].name == name) {
            return Node::set_output(i, std::move(val));
        }
    }
}


}
