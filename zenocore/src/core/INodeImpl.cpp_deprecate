#include <zeno/utils/helper.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/DictObject.h>
#include <zeno/core/INodeImpl.h>
#include <zeno/core/NodeImpl.h>
#include <zeno/utils/interfaceutil.h>


namespace zeno
{
    INodeImpl::INodeImpl(INode* pNode) : m_pImpl(nullptr) {
        m_pImpl = new NodeImpl(pNode);
        pNode->m_pAdapter = this;
    }

    INodeImpl::~INodeImpl() {
        delete m_pImpl;
    }

    CustomUI INodeImpl::export_customui() const {
        return m_pImpl->export_customui();
    }

    zany INodeImpl::get_input_object(const zeno::String& param) {
        return m_pImpl->get_input(zsString2Std(param));
    }

    zeno::SharedPtr<PrimitiveObject> INodeImpl::get_input_PrimitiveObject(const zeno::String& param) {
        return std::dynamic_pointer_cast<PrimitiveObject>(get_input_object(param));
    }

    zeno::SharedPtr<ListObject> INodeImpl::get_input_ListObject(const zeno::String& param) {
        return std::dynamic_pointer_cast<ListObject>(get_input_object(param));
    }

    zeno::SharedPtr<DictObject> INodeImpl::get_input_DictObject(const zeno::String& param) {
        return std::dynamic_pointer_cast<DictObject>(get_input_object(param));
    }

    void INodeImpl::set_output_object(const zeno::String& param, zany pObject) {
        m_pImpl->set_output(zsString2Std(param), pObject);
    }

    int INodeImpl::get_input2_int(const zeno::String& param) {
        return m_pImpl->get_input2<int>(zsString2Std(param));
    }

    int INodeImpl::get_param_int(const zeno::String& param) {
        return m_pImpl->get_param<int>(zsString2Std(param));
    }

    float INodeImpl::get_param_float(const zeno::String& param) {
        return m_pImpl->get_param<float>(zsString2Std(param));
    }

    String INodeImpl::get_param_string(const zeno::String& param) {
        return stdString2zs(m_pImpl->get_param<std::string>(zsString2Std(param)));
    }

    bool INodeImpl::get_param_bool(const zeno::String& param) {
        return m_pImpl->get_param<bool>(zsString2Std(param));
    }

    zany INodeImpl::get_output(const zeno::String& param) {
        return m_pImpl->get_output_obj(zsString2Std(param));
    }

    float INodeImpl::get_input2_float(const zeno::String& param) {
        return m_pImpl->get_input2<float>(zsString2Std(param));
    }

    zeno::String INodeImpl::get_input2_string(const zeno::String& param) {
        std::string val = m_pImpl->get_input2<std::string>(zsString2Std(param));
        return stdString2zs(val);
    }

    zeno::Vec2i INodeImpl::get_input2_vec2i(const zeno::String& param) {
        return toAbiVec2i(m_pImpl->get_input2<zeno::vec2i>(zsString2Std(param)));
    }

    zeno::Vec2f INodeImpl::get_input2_vec2f(const zeno::String& param) {
        return toAbiVec2f(m_pImpl->get_input2<zeno::vec2f>(zsString2Std(param)));
    }

    zeno::Vec3i INodeImpl::get_input2_vec3i(const zeno::String& param) {
        return toAbiVec3i(m_pImpl->get_input2<zeno::vec3i>(zsString2Std(param)));
    }

    zeno::Vec3f INodeImpl::get_input2_vec3f(const zeno::String& param) {
        return toAbiVec3f(m_pImpl->get_input2<zeno::vec3f>(zsString2Std(param)));
    }

    zeno::Vec4f INodeImpl::get_input2_vec4i(const zeno::String& param) {
        return toAbiVec4f(m_pImpl->get_input2<zeno::vec4f>(zsString2Std(param)));
    }

    zeno::Vec4i INodeImpl::get_input2_vec4f(const zeno::String& param) {
        return toAbiVec4i(m_pImpl->get_input2<zeno::vec4i>(zsString2Std(param)));
    }

    bool INodeImpl::get_input2_bool(const zeno::String& param) {
        return m_pImpl->get_input2<bool>(zsString2Std(param));
    }

    bool INodeImpl::has_input(const zeno::String& param) {
        return m_pImpl->has_input(zsString2Std(param));
    }

    void INodeImpl::set_output_int(const zeno::String& param, int val) {
        return m_pImpl->set_output2(zsString2Std(param), val);
    }

    void INodeImpl::set_output_float(const zeno::String& param, float val) {
        return m_pImpl->set_output2(zsString2Std(param), val);
    }

    void INodeImpl::set_output_vec2f(const zeno::String& param, Vec2f val) {
        return m_pImpl->set_output2(zsString2Std(param), toVec2f(val));
    }

    void INodeImpl::set_output_vec2i(const zeno::String& param, Vec2i val) {
        return m_pImpl->set_output2(zsString2Std(param), toVec2i(val));
    }

    void INodeImpl::set_output_vec3f(const zeno::String& param, Vec3f val) {
        return m_pImpl->set_output2(zsString2Std(param), toVec3f(val));
    }

    void INodeImpl::set_output_vec3i(const zeno::String& param, Vec3i val) {
        return m_pImpl->set_output2(zsString2Std(param), toVec3i(val));
    }

    int INodeImpl::GetFrameId() const {
        return m_pImpl->getGlobalState()->getFrameId();
    }

    GlobalState* INodeImpl::getGlobalState() {
        return m_pImpl->getGlobalState();
    }

}