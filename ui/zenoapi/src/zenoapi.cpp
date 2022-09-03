#include "zenoapi.h"
#include "zapplication_impl.h"

namespace zenoapi
{
    void openFile(const std::string& fn)
    {
        ZApplication_Impl::instance().openFile(fn);
    }

    std::string addNode(const std::string& subg, const std::string& nodeCls)
    {
        auto inst = ZApplication_Impl::instance();
        auto spGraph = inst.getSubgraph(subg);
        if (spGraph)
        {
            auto spNode = spGraph->addNode(nodeCls);
            if (spNode) {
                return spNode->getIdent();
            }
        }
        return "";
    }
}
