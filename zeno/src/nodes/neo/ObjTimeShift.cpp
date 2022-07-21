#include <zeno/zeno.h>
#include <zeno/types/ListObject.h>

namespace zeno {
namespace {

struct ObjTimeShift : INode {
    std::vector<std::shared_ptr<IObject>> m_objseq;

    virtual void apply() override {
        auto obj = get_input<IObject>("obj");
        auto offset = get_input2<int>("offset");
        std::shared_ptr<IObject> prevObj;
        auto &objseq = has_input("customList") ?
            get_input<ListObject>("customList")->arr : m_objseq;
        if (offset < 0) {
            objseq.resize(1);
            prevObj = std::move(objseq[0]);
            objseq[0] = obj->clone();
        } else {
            objseq.push_back(obj->clone());
            if (offset < objseq.size())
                prevObj = objseq[objseq.size() - 1 - offset];
            else
                prevObj = objseq[0];
        }
        set_output("obj", std::move(obj));
        set_output("prevObj", std::move(prevObj));
    }
};

ZENDEFNODE(ObjTimeShift, {
    {
    {"IObject", "obj"},
    {"int", "offset", "1"},
    {"ListObject", "customList"},
    },
    {
    {"IObject", "obj"},
    {"IObject", "prevObj"},
    },
    {
    },
    {"primitive"},
});

//struct ObjCacheToDisk : INode {
//};

}
}
