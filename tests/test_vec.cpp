#include <gtest/gtest.h>
#include "gtest_helpers.h"

ZENO_NAMESPACE_BEGIN
namespace tests {

class test_vec : public ::testing::TestWithParam<math::vec3f> {
protected:
};

TEST_P(test_vec, test_clamp) {
    math::vec3f x = GetParam();
    x = clamp(x, -0.5f, 1);
    EXPECT_EQ(x, min(max(x, math::vec3f(-0.5f)), math::vec3f(1.0f)));
}

INSTANTIATE_TEST_SUITE_P(
        test_clamp,
        test_vec,
        ::testing::Values
                ( math::vec3f(3.14f, -1.2f, 0.2f)
                , math::vec3f(1.f, -0.14f, -3.14f)
                , math::vec3f(0.f, 1.01f, -0.5f)
        ));

}
ZENO_NAMESPACE_END
