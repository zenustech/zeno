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
#include <zeno/extra/GlobalState.h>
#include <zeno/core/data.h>
#include <zeno/utils/uuid.h>
#include <functional>

namespace zeno {

struct Graph;
struct INodeClass;
struct Scene;
struct Session;
struct GlobalState;
struct TempNodeCaller;
struct IParam;

struct INode : std::enable_shared_from_this<INode> {
public:
    Graph *graph = nullptr;
    INodeClass *nodeClass = nullptr;

    std::string m_name;
    std::string m_nodecls;
    std::pair<float, float> m_pos;

    /*
    std::map<std::string, std::pair<std::string, std::string>> inputBounds;
    std::map<std::string, zany> inputs;
    std::map<std::string, zany> outputs;    //考虑到参数名字会被更改，放在这里需要调整key，麻烦
    */

    std::map<std::string, std::shared_ptr<IParam>> inputs_;
    std::map<std::string, std::shared_ptr<IParam>> outputs_;

    //std::set<std::string> kframes;
    //std::set<std::string> formulas;
    zany muted_output;

    NodeStatus m_status = NodeStatus::None;
    bool m_dirty = false;

    ZENO_API INode();
    ZENO_API virtual ~INode();

    //INode(INode const &) = delete;
    //INode &operator=(INode const &) = delete;
    //INode(INode &&) = delete;
    //INode &operator=(INode &&) = delete;

    ZENO_API void doComplete();
    ZENO_API void doApply();
    ZENO_API void doOnlyApply();
    ZENO_API zany resolveInput(std::string const& id);

    //BEGIN new api
    void init(const NodeData& dat);
    ZENO_API void set_input_defl(std::string const& name, zvariant defl);
    ZENO_API zvariant get_input_defl(std::string const& name);
    ZENO_API std::string get_nodecls() const;
    ZENO_API std::string get_ident() const;


    ZENO_API void set_view(bool bOn);
    ZENO_API bool is_view() const;
    ZENO_API void mark_dirty(bool bOn);
    ZENO_API bool is_dirty() const;

    ZENO_API virtual std::vector<std::shared_ptr<IParam>> get_input_params() const;
    ZENO_API virtual std::vector<std::shared_ptr<IParam>> get_output_params() const;
    ZENO_API std::shared_ptr<IParam> get_input_param(std::string const& name) const;
    ZENO_API std::shared_ptr<IParam> get_output_param(std::string const& name) const;

    ZENO_API bool update_param(const std::string& name, const zvariant& new_value);
    CALLBACK_REGIST(update_param, void, const std::string&, zvariant, zvariant)

    ZENO_API virtual params_change_info update_editparams(const ParamsUpdateInfo& params);

    ZENO_API void set_name(const std::string& name);
    ZENO_API std::string get_name() const;

    ZENO_API void set_pos(std::pair<float, float> pos);
    CALLBACK_REGIST(set_pos, void, std::pair<float, float>)
    ZENO_API std::pair<float, float> get_pos() const;

    ZENO_API void set_status(NodeStatus status);
    CALLBACK_REGIST(set_status, void, NodeStatus)
    ZENO_API NodeStatus get_status() const;

    //END new api

    void add_input_param(std::shared_ptr<IParam> param);
    void add_output_param(std::shared_ptr<IParam> param);
    void directly_setinputs(std::map<std::string, zany> inputs);
    std::map<std::string, zany> getoutputs();

protected:
    ZENO_API virtual void complete();
    ZENO_API virtual void apply() = 0;
    ZENO_API std::vector<std::pair<std::string, zany>> getinputs();
    ZENO_API std::vector<std::pair<std::string, zany>> getoutputs2();
    ZENO_API std::pair<std::string, std::string> getinputbound(std::string const& name, std::string const& msg = "") const;

private:
    zany process(std::shared_ptr<IParam> in_param);
    float resolve(const std::string& formulaOrKFrame, const ParamType type);
    template<class T, class E> zany resolveVec(const zvariant& defl, const ParamType type);

public:
    //为名为ds的输入参数，求得这个参数在依赖边的求值下的值，或者没有依赖边下的默认值。
    ZENO_API bool requireInput(std::string const &ds);
    ZENO_API bool requireInput(std::shared_ptr<IParam> param);

    //preApply是先解决所有输入参数的求值问题，再调用apply执行具体算法。
    ZENO_API virtual void preApply();

    ZENO_API Graph *getThisGraph() const;
    ZENO_API Session *getThisSession() const;
    ZENO_API GlobalState *getGlobalState() const;

    ZENO_API bool has_input(std::string const &id) const;
    ZENO_API bool set_input(std::string const& name, zany obj);
    ZENO_API zany get_input(std::string const &id) const;
    ZENO_API bool has_output(std::string const& name) const;
    ZENO_API bool set_output(std::string const &id, zany obj);
    ZENO_API zany get_output(std::string const& sock_name);

    ZENO_API bool has_keyframe(std::string const &id) const;
    ZENO_API zany get_keyframe(std::string const &id) const;

    ZENO_API bool has_formula(std::string const &id) const;
    ZENO_API zany get_formula(std::string const &id) const;

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
    [[deprecated("use get_input2<T>(id + ':')")]]
    T get_param(std::string const &id) const {
        return get_input2<T>(id + ':');
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

};

}
