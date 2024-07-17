#include "data.h"
#include "test.h"
#include "reflect/type"
#include "reflect/container/any"
#include "reflect/reflection.generated.hpp"

using namespace zeno::reflect;

int main() {

    Any soo = make_any<Soo>();
    Any param1 = make_any<int>(123);

    TypeHandle handle = get_type<Soo>();

    // param1 with type int will automaticly convert to int*
    handle->get_member_functions()[0]->invoke(soo, { param1 });

    // Create an Any holding a pointer on stack.
    int a = 666;
    param1 = make_any<int*>(&a);

    // Call it with int*
    handle->get_member_functions()[0]->invoke(soo, { param1 });

    return 0;
}
