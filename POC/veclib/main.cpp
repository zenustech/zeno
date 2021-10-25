#include "vec_type.h"
#include "vec_operators.h"


using namespace zeno::ztd;


int main() {
    vec<3, float> x;
    x += 1;
    auto [a, b, c] = x;
    return 0;
}
