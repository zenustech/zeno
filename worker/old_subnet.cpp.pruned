#include <zeno/dop/dop.h>
#include <iostream>

int main()
{
    auto subnet = std::make_unique<dop::SubnetNode>();
    auto twiceInt = subnet->addNode(dop::descriptor_table().at("TwiceInt"));
    twiceInt->linkInput(0, subnet->subnetIn.get(), 0);
    subnet->subnetOut->linkInput(0, twiceInt, 0);

    subnet->setInput(0, ztd::make_any(42));
    subnet->apply();
    std::cout << value_cast<int>(subnet->getOutput(0)) << std::endl;

    return 0;
}
