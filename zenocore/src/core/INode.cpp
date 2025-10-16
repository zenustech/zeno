#include <zeno/core/INode.h>
#include <zeno/core/NodeImpl.h>
#include <zeno/utils/interfaceutil.h>
#include <zeno/types/IGeometryObject.h>

namespace zeno
{
    CustomUI INode::export_customui() const {
        return CustomUI();
    }

    NodeType INode::type() const {
        return Node_Normal;
    }

    zeno::String INode::uuid() const {
        return stdString2zs(m_pAdapter->get_uuid());
    }

    IObject* INode::get_input(const zeno::String& param) {
        return m_pAdapter->get_input_obj(zsString2Std(param));
    }

    zany INode::clone_input(const zeno::String& param) {
        return m_pAdapter->clone_input(zsString2Std(param));
    }

    std::unique_ptr<PrimitiveObject> INode::clone_input_PrimitiveObject(const zeno::String& param) {
        auto rawPtr = m_pAdapter->get_input_obj(zsString2Std(param));
        if (dynamic_cast<PrimitiveObject*>(rawPtr)) {
            return std::unique_ptr<PrimitiveObject>(static_cast<PrimitiveObject*>(clone_input(param).release()));
        }
        return nullptr;
    }

    PrimitiveObject* INode::get_input_PrimitiveObject(const zeno::String& param) {
        auto rawPtr = get_input(param);
        if (dynamic_cast<PrimitiveObject*>(rawPtr))
            return static_cast<PrimitiveObject*>(rawPtr);
        return nullptr;
    }

    std::unique_ptr<GeometryObject_Adapter> INode::clone_input_Geometry(const zeno::String& param) {
        auto rawPtr = m_pAdapter->get_input_obj(zsString2Std(param));
        if (dynamic_cast<GeometryObject_Adapter*>(rawPtr))
            return std::unique_ptr<GeometryObject_Adapter>(static_cast<GeometryObject_Adapter*>(clone_input(param).release()));
        return nullptr;
    }

    GeometryObject_Adapter* INode::get_input_Geometry(const zeno::String& param) {
        auto rawPtr = get_input(param);
        if (dynamic_cast<GeometryObject_Adapter*>(rawPtr))
            return static_cast<GeometryObject_Adapter*>(rawPtr);
        return nullptr;
    }

    ListObject* INode::get_input_ListObject(const zeno::String& param) {
        return dynamic_cast<ListObject*>(get_input(param));
    }

    DictObject* INode::get_input_DictObject(const zeno::String& param) {
        return dynamic_cast<DictObject*>(get_input(param));
    }

    int INode::get_input2_int(const zeno::String& param) {
        const auto& anyVal = m_pAdapter->get_param_result(zsString2Std(param));
        int res = 0;
        if (anyVal.type().hash_code() == zeno::types::gParamType_Int) {
            res = zeno::reflect::any_cast<int>(anyVal);
        }
        else if (anyVal.type().hash_code() == zeno::types::gParamType_Float) {
            res = zeno::reflect::any_cast<float>(anyVal);
        }
        else if (anyVal.type().hash_code() == zeno::types::gParamType_Bool) {
            res = zeno::reflect::any_cast<bool>(anyVal);
        }
        return res;
    }

    float INode::get_input2_float(const zeno::String& param) {
        const auto& anyVal = m_pAdapter->get_param_result(zsString2Std(param));
        float res = 0;
        if (anyVal.type().hash_code() == zeno::types::gParamType_Int) {
            res = zeno::reflect::any_cast<int>(anyVal);
        }
        else if (anyVal.type().hash_code() == zeno::types::gParamType_Float) {
            res = zeno::reflect::any_cast<float>(anyVal);
        }
        else if (anyVal.type().hash_code() == zeno::types::gParamType_Bool) {
            res = zeno::reflect::any_cast<bool>(anyVal);
        }
        return res;
    }

    String INode::get_input2_string(const zeno::String& param) {
        return stdString2zs(any_cast<std::string>(m_pAdapter->get_param_result(zsString2Std(param))));
    }

    zeno::Vec2i INode::get_input2_vec2i(const zeno::String& param) {
        return toAbiVec2i(any_cast<vec2i>(m_pAdapter->get_param_result(zsString2Std(param))));
    }

    zeno::Vec2f INode::get_input2_vec2f(const zeno::String& param) {
        return toAbiVec2f(any_cast<vec2f>(m_pAdapter->get_param_result(zsString2Std(param))));
    }

    zeno::Vec3i INode::get_input2_vec3i(const zeno::String& param) {
        return toAbiVec3i(any_cast<vec3i>(m_pAdapter->get_param_result(zsString2Std(param))));
    }

    zeno::Vec3f INode::get_input2_vec3f(const zeno::String& param) {
        return toAbiVec3f(any_cast<vec3f>(m_pAdapter->get_param_result(zsString2Std(param))));
    }

    zeno::Vec4i INode::get_input2_vec4i(const zeno::String& param) {
        return toAbiVec4i(any_cast<vec4i>(m_pAdapter->get_param_result(zsString2Std(param))));
    }

    zeno::Vec4f INode::get_input2_vec4f(const zeno::String& param) {
        return toAbiVec4f(any_cast<vec4f>(m_pAdapter->get_param_result(zsString2Std(param))));
    }

    bool INode::get_input2_bool(const zeno::String& param) {
        return any_cast<bool>(m_pAdapter->get_param_result(zsString2Std(param)));
    }

    bool INode::has_input(const zeno::String& param) {
        return m_pAdapter->has_input(zsString2Std(param));
    }

    bool INode::has_link_input(const zeno::String& param) {
        return m_pAdapter->has_link_input(zsString2Std(param));
    }

    bool INode::is_upstream_dirty(const zeno::String& param) const {
        return m_pAdapter->is_upstream_dirty(zsString2Std(param));
    }

    void INode::check_break() {
        m_pAdapter->check_break_and_return();
    }

    void INode::clearCalcResults() {

    }

    float INode::time() const {
        return 1.0f;
    }

    bool INode::set_output(const zeno::String& param, zany&& pObject) {
        return m_pAdapter->set_output(zsString2Std(param), std::move(pObject));
    }

    int INode::get_param_int(const zeno::String& param) {
        return get_input2_int(param);
    }

    float INode::get_param_float(const zeno::String& param) {
        return get_input2_float(param);
    }

    String INode::get_param_string(const zeno::String& param) {
        return get_input2_string(param);
    }

    bool INode::get_param_bool(const zeno::String& param) {
        return get_input2_bool(param);
    }

    bool INode::set_output_int(const zeno::String& param, int val) {
        return m_pAdapter->set_primitive_output(zsString2Std(param), val);
    }

    bool INode::set_output_float(const zeno::String& param, float val) {
        return m_pAdapter->set_primitive_output(zsString2Std(param), val);
    }

    bool INode::set_output_string(const zeno::String& param, zeno::String val) {
        return m_pAdapter->set_primitive_output(zsString2Std(param), zsString2Std(val));
    }

    bool INode::set_output_vec2f(const zeno::String& param, Vec2f val) {
        return m_pAdapter->set_primitive_output(zsString2Std(param), toVec2f(val));
    }

    bool INode::set_output_vec2i(const zeno::String& param, Vec2i val) {
        return m_pAdapter->set_primitive_output(zsString2Std(param), toVec2i(val));
    }

    bool INode::set_output_vec3f(const zeno::String& param, Vec3f val) {
        return m_pAdapter->set_primitive_output(zsString2Std(param), toVec3f(val));
    }

    bool INode::set_output_vec3i(const zeno::String& param, Vec3i val) {
        return m_pAdapter->set_primitive_output(zsString2Std(param), toVec3i(val));
    }

    int INode::GetFrameId() const {
        return m_pAdapter->getGlobalState()->getFrameId();
    }

    GlobalState* INode::getGlobalState() {
        return m_pAdapter->getGlobalState();
    }


}