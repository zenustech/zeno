#include "ZFX.h"
#include "x64/Program.h"

static zfx::Compiler<zfx::x64::Program> compiler;

int main() {
    std::string code("tmp = @pos + 0.5\n@pos = tmp + 3.14 * tmp + 2.718 / (@pos * tmp + 1)");
    auto func = [](float pos) -> float {
        auto tmp = pos + 0.5f;
        pos = tmp + 3.14f * tmp + 2.718f / (pos * tmp + 1);
        return pos;
    };

    std::map<std::string, int> symdims;
    symdims["@pos"] = 3;

    auto prog = compiler.compile(code, symdims);

    float arr[4] = {1, 2, 3, 4};
    float arr2[4] = {0, 0, 0, 0};
    float arr3[4] = {0, 0, 0, 0};

    printf("expected:");
    for (auto val: arr) {
        val = func(val);
        printf(" %f", val);
    }
    printf("\n");

    prog->set_channel_pointer("@pos", 0, arr);
    prog->set_channel_pointer("@pos", 1, arr2);
    prog->set_channel_pointer("@pos", 2, arr3);
    prog->execute();

    printf("result:");
    for (auto val: arr) {
        printf(" %f", val);
    }
    printf("\n");

    return 0;
}
