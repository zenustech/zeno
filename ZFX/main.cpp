#include "ZFX.h"
#include "x64/Program.h"

int main() {
    std::string code("pos = pos + 0.5");
    auto prog = zfx::compile_to<zfx::x64::Program>(code);

    float arr[4] = {1, 2, 3, 4};
    prog->set_channel_pointer("pos", arr);
    prog->execute();

    printf("result:");
    for (auto val: arr) printf(" %f", val);
    printf("\n");

    return 0;
}
