#include "vec_type.h"
#include "vec_operators.h"
#include "vec_functions.h"


using namespace zeno::ztd;


int main() {
    vec<3, float> x;
    x += 1;
    auto [a, b, c] = x;
    max(x, x);
    return 0;
}
