#include <gtest/gtest.h>
#include "gtest_helpers.h"

namespace zs::tests {

class test_vec : public testing::TestWithParam<ztd::vec3f> {
protected:
};

TEST_P(test_vec, test_clamp) {
    ztd::vec3f x = GetParam();
    x = clamp(x, -0.5f, 1);
    EXPECT_TRUE(
            vall(x == min(max(x, ztd::vec3f(-0.5f)), ztd::vec3f(1.0f)))
            ) << SHOW_VAR(x);
}

INSTANTIATE_TEST_SUITE_P(
        test_clamp,
        test_vec,
        testing::Values
                ( ztd::vec3f(3.14f, -1.2f, 0.2f)
                , ztd::vec3f(1.f, -0.14f, -3.14f)
                , ztd::vec3f(0.f, 1.01f, -0.5f)
        ));

}
