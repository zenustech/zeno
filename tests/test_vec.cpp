#include <gtest/gtest.h>
#include <zeno2/ztd/vec.h>

namespace zeno2 {

TEST(test_vec, test_clamp) {
    ztd::vec3f x;
    x = ztd::clamp(x, 0, 1);
}

}
