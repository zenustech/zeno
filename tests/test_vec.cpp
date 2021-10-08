#include <gtest/gtest.h>
#include <zeno2/ztd/vec.h>

namespace zeno2 {

TEST(test_vec, test_clamp) {
    ztd::vec3f x(3.14f, -1.2f, 0.2f);
    x = clamp(x, -0.5f, 1);
    EXPECT_TRUE(vall(x == ztd::vec3f(1.0f, -0.5f, 0.2f)));
}

}
