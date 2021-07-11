#include "ZFX.h"
#include "x64/Program.h"

static zfx::Compiler<zfx::x64::Program> compiler;

int main() {
    std::string code("tmp = pos + 0.5\npos = tmp + 2");
    std::map<std::string, int> symdims;
    symdims["pos"] = 1;

    auto prog = compiler.compile(code, symdims);

    float arr[4] = {1, 2, 3, 4};
    prog->set_channel_pointer("pos", 0, arr);
    prog->execute();

    printf("result:");
    for (auto val: arr) printf(" %f", val);
    printf("\n");  // 3.5 4.5 5.5 6.5

    return 0;
}
