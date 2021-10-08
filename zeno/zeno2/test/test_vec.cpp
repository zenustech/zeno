#include <zeno2/ztd/vec.h>

namespace zeno2 {

int test_main() {
    ztd::vec3f x;
    x = ztd::clamp(x, 0, 1);
    return 0;
}

}
