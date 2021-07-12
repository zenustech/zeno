#include "ZFX.h"
#include "x64/Program.h"
#include <cmath>

static zfx::Compiler<zfx::x64::Program> compiler;

int main() {
    std::string code("@pos = sqrt(@pos.y)");
    auto func = [](float pos) -> float {
        return std::sqrt(pos);
    };

    std::map<std::string, int> symdims;
    symdims["@pos"] = 2;

    auto prog = compiler.compile(code, symdims);

    float arr[4] = {1, 2, 3, 4};
    float arr2[4] = {2, 3, 4, 5};

    printf("expected:");
    for (auto val: arr) {
        val = func(val);
        printf(" %f", val);
    }
    printf("\n");

    prog->set_channel_pointer("@pos", 0, arr);
    prog->set_channel_pointer("@pos", 1, arr2);
    prog->execute();

    printf("result:");
    for (auto val: arr) {
        printf(" %f", val);
    }
    printf("\n");

    return 0;
}
