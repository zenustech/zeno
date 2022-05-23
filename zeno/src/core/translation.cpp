#include <zeno/core/Session.h>
#include <zeno/utils/Translator.h>

namespace zeno {
namespace {

static int helper = ([]{

    getSession().translatorNodeName->load(R"(
ToView=ToView
BindMaterial=绑定材质
)" R"(
)");

    getSession().translatorSocketName->load(R"(
object=对象
prim=图元
size=尺寸
)" R"(
)");

}(), 0);

}
}
