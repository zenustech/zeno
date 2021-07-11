#include "ZFX.h"
#include "x64/Program.h"

int main() {
    using namespace zfx;

    std::string code("pos = pos + 0.5");
    auto 
        [ assem
        , symbols
        ] = compile_to_assembly
        ( code
        );

    auto prog = x64::assemble_program(assem);

    float arr[4] = {1, 2, 3, 4};
    prog->set_channel_pointer(0, arr);
    prog->execute();

    printf("result:");
    for (auto val: arr) printf(" %f", val);
    printf("\n");

    return 0;
}
