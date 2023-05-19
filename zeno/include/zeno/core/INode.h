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

namespace zeno {

struct Graph;
struct INodeClass;
struct Scene;
struct Session;
struct GlobalState;
struct TempNodeCaller;

struct INode {
public:
    Graph *graph = nullptr;
    INodeClass *nodeClass = nullptr;

    std::string myname;
    std::map<std::string, std::pair<std::string, std::string>> inputBounds;
    std::map<std::string, zany> inputs;
    std::map<std::string, zany> outputs;
    std::set<std::string> kframes;
    zany muted_output;

    ZENO_API INode();
    ZENO_API virtual ~INode();

    //INode(INode const &) = delete;
    //INode &operator=(INode const &) = delete;
    //INode(INode &&) = delete;
    //INode &operator=(INode &&) = delete;

    ZENO_API void doComplete();
    ZENO_API void doApply();
    ZENO_API void doOnlyApply();

protected:
    ZENO_API bool requireInput(std::string const &ds);

    ZENO_API virtual void preApply();
    ZENO_API virtual void complete();
    ZENO_API virtual void apply() = 0;

    ZENO_API Graph *getThisGraph() const;
    ZENO_API Session *getThisSession() const;
    ZENO_API GlobalState *getGlobalState() const;

    ZENO_API bool has_input(std::string const &id) const;
    ZENO_API zany get_input(std::string const &id) const;
    ZENO_API void set_output(std::string const &id, zany obj);

    ZENO_API bool has_keyframe(std::string const &id) const;

    template <class T>
    std::shared_ptr<T> get_input(std::string const &id) const {
        auto obj = get_input(id);
        return safe_dynamic_cast<T>(std::move(obj), "input socket `" + id + "` of node `" + myname + "`");
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

    auto calculateKeyFrame(std::string const &id) const
    {
        auto curves = dynamic_cast<zeno::CurveObject*>(get_input(id).get());
        auto value = get_input(id);
        int frame = getGlobalState()->frameid;
        if (curves->keys.size() == 1) {
            auto val = curves->keys.begin()->second.eval(frame);
            value = objectFromLiterial(val);
        } else {
            int size = curves->keys.size();
            if (size == 2) {
                zeno::vec2f vec2;
                for (std::map<std::string, CurveData>::const_iterator it = curves->keys.cbegin();
                     it != curves->keys.cend(); it++) {
                    int index = it->first == "x" ? 0 : 1;
                    vec2[index] = it->second.eval(frame);
                }
                value = objectFromLiterial(vec2);
            }
            else if (size == 3) {
                zeno::vec3f vec3;
                for (std::map<std::string, CurveData>::const_iterator it = curves->keys.cbegin();
                     it != curves->keys.cend(); it++) {
                    int index = it->first == "x" ? 0 : it->first == "y" ? 1 : 2;
                    vec3[index] = it->second.eval(frame);
                }
                value = objectFromLiterial(vec3);
            }
            else if (size == 4) {
                zeno::vec4f vec4;
                for (std::map<std::string, CurveData>::const_iterator it = curves->keys.cbegin();
                     it != curves->keys.cend(); it++) {
                    int index = it->first == "x" ? 0 : it->first == "y" ? 1 : it->first == "z" ? 2 : 3;
                    vec4[index] = it->second.eval(frame);
                }
                value = objectFromLiterial(vec4);
            }
        }
        return value;
    }

    template <class T>
    auto get_input2(std::string const &id) const {
        if (has_keyframe(id)) {
            auto value = calculateKeyFrame(id);
            return objectToLiterial<T>(value, "input socket `" + id + "` of node `" + myname + "`");
        }
        return objectToLiterial<T>(get_input(id), "input socket `" + id + "` of node `" + myname + "`");
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
