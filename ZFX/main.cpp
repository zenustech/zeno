#include "ZFX.h"
#include "x64/Program.h"

static zfx::Compiler<zfx::x64::Program> compiler;

int main() {
    std::string code("pos = pos + 0.5");
    auto prog = compiler.compile(code);

    float arr[4] = {1, 2, 3, 4};
    prog->set_channel_pointer("pos", arr);
    prog->execute();

    printf("result:");
    for (auto val: arr) printf(" %f", val);
    printf("\n");

    return 0;
}
