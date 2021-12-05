#include <zeno/dop/dop.h>
#include <fstream>

int main()
{
    dop::SubnetNode sub;
    auto n = sub.addNode(dop::descriptor_table().at("PrintInt"))
        ->setInput(0, ztd::make_any(42))
        ;
    sub.subnetOut->setInput(0, n, 0);
    sub.apply();

    return 0;
}
