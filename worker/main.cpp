#include <zeno/dop/dop.h>
#include <fstream>

int main()
{
    dop::SubnetNode sub;
    auto n = sub.addNode(dop::descriptor_table().at("SleepFor"));
    n->inputs.at(0).value = ztd::make_any(1000);
    n->apply();

    return 0;
}
