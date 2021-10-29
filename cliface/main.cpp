#include <zeno/dop/dop.h>

USING_ZENO_NAMESPACE



int main()
{
    auto n1 = dop::desc_of("ReadOBJMesh").factory();
    n1->inputs.at(0) = dop::Input_Value{"hello"};
    n1->apply();
    n1->outputs.at(0);

    return 0;
}
