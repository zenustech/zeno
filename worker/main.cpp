#include <zeno/dop/dop.h>
#include <iostream>

int main()
{
    dop::SubnetNode sub;
    auto n = sub.addNode(dop::descriptor_table().at("TwiceInt"))
        ->linkInput(0, sub.subnetIn.get(), 0)
        ;
    sub.subnetOut->linkInput(0, n, 0);

    sub.setInput(0, ztd::make_any(42));
    sub.apply();
    std::cout << value_cast<int>(sub.getOutput(0)) << std::endl;

    return 0;
}
