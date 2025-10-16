#include <zeno/zeno.h>
#include <zeno/types/ListObject_impl.h>

namespace zeno {
namespace {
#if 0
struct ObjTimeShift : INode {
    std::vector<zany> m_objseq;

    virtual void apply() override {
        auto obj = ZImpl(get_input<IObject>("obj"));
        auto offset = ZImpl(get_input2<int>("offset"));
        zany prevObj;
        auto objseq = ZImpl(has_input("customList")) ?
            ZImpl(get_input<ListObject>("customList"))->m_impl->get() : m_objseq;
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
        ZImpl(set_output("obj", std::move(obj)));
        ZImpl(set_output("prevObj", std::move(prevObj)));
    }
};

ZENDEFNODE(ObjTimeShift, {
    {
    {"IObject", "obj"},
    {gParamType_Int, "offset", "1"},
    {gParamType_List, "customList"},
    },
    {
    {"IObject", "obj"},
    {"IObject", "prevObj"},
    },
    {
    },
    {"primitive"},
});
#endif

//struct ObjCacheToDisk : INode {
//};

}
}
