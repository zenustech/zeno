#include <z2/ztd/vec.h>

namespace z2 {

int test_main() {
    ztd::vec3f x;
    x = ztd::clamp(x, 0, 1);
    return 0;
}

}
