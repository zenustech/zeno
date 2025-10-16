#include <zeno/formula/syntax_tree.h>

namespace zeno
{
    namespace zfx
    {
        ZfxVariable callFunction(const std::string& name, const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext);
    }
}