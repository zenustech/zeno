#pragma once

#include <zeno/utils/api.h>
#include <zeno/core/IObject.h>
#include <zeno/utils/safe_dynamic_cast.h>
#include <zeno/funcs/LiterialConverter.h>
#include <variant>
#include <memory>
#include <string>
#include <set>
#include <map>
#include <zeno/types/CurveObject.h>
#include <zeno/types/DictObject.h>
#include <zeno/types/ListObject.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/core/data.h>
#include <zeno/utils/uuid.h>
#include <zeno/core/CoreParam.h>
#include <functional>


namespace zeno {

struct Graph;
struct INodeClass;
struct Scene;
struct Session;
struct GlobalState;
struct TempNodeCaller;
struct CoreParam;
struct ObjectParam;
struct PrimitiveParam;
struct ObjectLink;
struct PrimitiveLink;
struct SubnetNode;

namespace reflect {
    class TypeHandle;
}


class INode : public std::enable_shared_from_this<INode>
{
public:
    INodeClass *nodeClass = nullptr;

    zany muted_output;

    ZENO_API INode();
    ZENO_API virtual ~INode();
    ZENO_API zeno::reflect::TypeHandle* getReflectType();

    ZENO_API void doComplete();
    ZENO_API void doApply();
    ZENO_API void doOnlyApply();

    //BEGIN new api
    ZENO_API void init(const NodeData& dat);
    ZENO_API std::string get_nodecls() const;
    ZENO_API std::string get_ident() const;
    ZENO_API std::string get_show_name() const;
    ZENO_API std::string get_show_icon() const;
    ZENO_API virtual CustomUI get_customui() const;
    ZENO_API ObjPath get_path() const;
    ZENO_API ObjPath get_uuid_path() const { return m_uuidPath; }
    ZENO_API std::string get_uuid() const;
    ZENO_API std::weak_ptr<Graph> getGraph() const { return graph; }
    void initUuid(std::shared_ptr<Graph> pGraph, const std::string nodecls);

    ZENO_API void set_view(bool bOn);
    CALLBACK_REGIST(set_view, void, bool)
    ZENO_API bool is_view() const;

    ZENO_API void mark_dirty(bool bOn, bool bWholeSubnet = true, bool bRecursively = true);
    ZENO_API bool is_dirty() const { return m_dirty; }
    ZENO_API NodeRunStatus get_run_status() const { return m_status; }

    ZENO_API ObjectParams get_input_object_params() const;
    ZENO_API ObjectParams get_output_object_params() const;
    ZENO_API PrimitiveParams get_input_primitive_params() const;
    ZENO_API PrimitiveParams get_output_primitive_params() const;
    ZENO_API ParamPrimitive get_input_prim_param(std::string const& name, bool* pExist = nullptr) const;
    ZENO_API ParamObject get_input_obj_param(std::string const& name, bool* pExist = nullptr) const;
    ZENO_API ParamPrimitive get_output_prim_param(std::string const& name, bool* pExist = nullptr) const;
    ZENO_API ParamObject get_output_obj_param(std::string const& name, bool* pExist = nullptr) const;

    ZENO_API std::string get_viewobject_output_param() const;
    ZENO_API virtual NodeData exportInfo() const;
    ZENO_API void set_result(bool bInput, const std::string& name, zany spObj);

    ZENO_API bool update_param(const std::string& name, const zvariant& new_value);
    CALLBACK_REGIST(update_param, void, const std::string&, zvariant, zvariant)

   ZENO_API bool update_param_socket_type(const std::string& name, SocketType type);
    CALLBACK_REGIST(update_param_socket_type, void, const std::string&, SocketType)

   ZENO_API bool update_param_type(const std::string& name, bool bPrim, ParamType type);
   CALLBACK_REGIST(update_param_type, void, const std::string&, ParamType)

   ZENO_API bool update_param_control(const std::string& name, ParamControl control);
   CALLBACK_REGIST(update_param_control, void, const std::string&, ParamControl)

   ZENO_API bool update_param_control_prop(const std::string& name, ControlProperty props);
   CALLBACK_REGIST(update_param_control_prop, void, const std::string&, ControlProperty)

   ZENO_API bool update_param_visible(const std::string& name, bool bVisible);
   CALLBACK_REGIST(update_param_visible, void, const std::string&, bool)

    ZENO_API virtual params_change_info update_editparams(const ParamsUpdateInfo& params);

    ZENO_API void set_name(const std::string& name);
    ZENO_API std::string get_name() const;

    ZENO_API void set_pos(std::pair<float, float> pos);
    CALLBACK_REGIST(set_pos, void, std::pair<float, float>)
    ZENO_API std::pair<float, float> get_pos() const;

    ZENO_API bool in_asset_file() const;

    void onInterrupted();
    void mark_previous_ref_dirty();

    //END new api
    bool add_input_prim_param(ParamPrimitive param);
    bool add_input_obj_param(ParamObject param);
    bool add_output_prim_param(ParamPrimitive param);
    bool add_output_obj_param(ParamObject param);
    void init_object_link(bool bInput, const std::string& paramname, std::shared_ptr<ObjectLink> spLink);
    void init_primitive_link(bool bInput, const std::string& paramname, std::shared_ptr<PrimitiveLink> spLink);
    bool isPrimitiveType(bool bInput, const std::string& param_name, bool& bExist);
    std::vector<EdgeInfo> getLinks() const;
    std::vector<EdgeInfo> getLinksByParam(bool bInput, const std::string& param_name) const;
    bool updateLinkKey(bool bInput, const std::string& param_name, const std::string& oldkey, const std::string& newkey);
    bool moveUpLinkKey(bool bInput, const std::string& param_name, const std::string& key);
    bool removeLink(bool bInput, const EdgeInfo& edge);
    zvariant resolveInput(std::string const& id);
    void mark_dirty_objs();
    std::vector<std::string> getWildCardParams(const std::string& name, bool bPrim);

protected:
    ZENO_API virtual void complete();
    //preApply是先解决所有输入参数（依赖）的求值问题
    ZENO_API virtual void preApply();
    ZENO_API virtual void apply();
    ZENO_API virtual void registerObjToManager();
    ZENO_API virtual void initParams(const NodeData& dat);
    ZENO_API bool set_primitive_input(std::string const& id, const zvariant& val);
    ZENO_API bool set_primitive_output(std::string const& id, const zvariant& val);

private:
    zvariant processPrimitive(PrimitiveParam* in_param);
    std::shared_ptr<DictObject> processDict(ObjectParam* in_param);
    std::shared_ptr<ListObject> processList(ObjectParam* in_param);
    bool receiveOutputObj(ObjectParam* in_param, zany outputObj);
    void reportStatus(bool bDirty, NodeRunStatus status);

    float resolve(const std::string& formulaOrKFrame, const ParamType type);
    template<class T, class E> T resolveVec(const zvariant& defl, const ParamType type);

public:
    //为名为ds的输入参数，求得这个参数在依赖边的求值下的值，或者没有依赖边下的默认值。
    ZENO_API bool requireInput(std::string const &ds);

    ZENO_API std::shared_ptr<Graph> getThisGraph() const;
    ZENO_API Session *getThisSession() const;
    ZENO_API GlobalState *getGlobalState() const;

    ZENO_API bool has_input(std::string const &id) const;
    ZENO_API zany get_input(std::string const &id) const;
    ZENO_API bool set_output(std::string const &id, zany obj);
    ZENO_API zany get_output_obj(std::string const& sock_name);

    template <class T>
    std::shared_ptr<T> get_input(std::string const &id) const {
        auto obj = get_input(id);
        return safe_dynamic_cast<T>(std::move(obj), "input socket `" + id + "` of node `" + m_name + "`");
    }

    template <class T>
    bool has_input(std::string const &id) const {
        if (!has_input(id)) return false;
        auto obj = get_input(id);
        return !!dynamic_cast<T *>(obj.get());
    }

    template <class T>
    bool has_input2(std::string const &id) const {
        if (!has_input(id)) return false;
        return objectIsLiterial<T>(get_input(id));
    }

    template <class T>
    auto get_input2(std::string const &id) const {
        return objectToLiterial<T>(get_input(id), "input socket `" + id + "` of node `" + m_name + "`");
    }

    template <class T>
    void set_output2(std::string const &id, T &&value) {
        set_output(id, objectFromLiterial(std::forward<T>(value)));
    }

    template <class T>
    [[deprecated("use get_input2<T>(id)")]]
    T get_param(std::string const &id) const {
        return get_input2<T>(id);
    }

    //[[deprecated("use get_param<T>")]]
    //ZENO_API std::variant<int, float, std::string> get_param(std::string const &id) const;

    template <class T = IObject>
    std::shared_ptr<T> get_input(std::string const &id, std::shared_ptr<T> const &defl) const {
        return has_input(id) ? get_input<T>(id) : defl;
    }

    template <class T>
    T get_input2(std::string const &id, T const &defl) const {
        return has_input(id) ? get_input2<T>(id) : defl;
    }

    ZENO_API TempNodeCaller temp_node(std::string const &id);

private:
    std::string m_name;
    std::string m_nodecls;
    std::string m_uuid;
    std::pair<float, float> m_pos;

    std::map<std::string, ObjectParam> m_inputObjs;
    std::map<std::string, PrimitiveParam> m_inputPrims;
    std::map<std::string, PrimitiveParam> m_outputPrims;
    std::map<std::string, ObjectParam> m_outputObjs;

    ObjPath m_uuidPath;
    NodeRunStatus m_status = Node_DirtyReadyToRun;
    std::weak_ptr<Graph> graph;
    bool m_bView = false;
    bool m_dirty = true;

    friend class SubnetNode;
};

}
