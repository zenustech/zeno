#include "utest.h"
#include <mcut/mcut.h>

#define NUM_ROUNDING_MODES 4

McRoundingModeFlags modes[] = {
    MC_ROUNDING_MODE_TO_NEAREST,
    MC_ROUNDING_MODE_TOWARD_ZERO,
    MC_ROUNDING_MODE_TOWARD_POS_INF,
    MC_ROUNDING_MODE_TOWARD_NEG_INF
};

struct SetRoundingMode {
    McContext context_;
    McRoundingModeFlags mode;
};

UTEST_I_SETUP(SetRoundingMode)
{
    if (utest_index < NUM_ROUNDING_MODES) {
        McResult err = mcCreateContext(&utest_fixture->context_, MC_DEBUG);
        ASSERT_TRUE(utest_fixture->context_ != nullptr);
        ASSERT_EQ(err, MC_NO_ERROR);
        utest_fixture->mode = modes[utest_index];
    }
}

UTEST_I_TEARDOWN(SetRoundingMode)
{
    if (utest_index < NUM_ROUNDING_MODES) {
        McResult err = mcReleaseContext(utest_fixture->context_);
        EXPECT_EQ(err, MC_NO_ERROR);
    }
}

UTEST_I(SetRoundingMode, setRoundingMode, NUM_ROUNDING_MODES)
{
    uint32_t roundingModeValSet = MC_ROUNDING_MODE_TO_NEAREST;
    EXPECT_EQ(mcSetRoundingMode(utest_fixture->context_, roundingModeValSet), MC_NO_ERROR);

    uint32_t roundingMode = 0;
    EXPECT_EQ(mcGetRoundingMode(utest_fixture->context_, &roundingMode), MC_NO_ERROR);

    ASSERT_EQ(roundingModeValSet, roundingMode);
}
