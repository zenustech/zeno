#include "LowerAST.h"
#include "Visitors.h"
#include "x64/Program.h"
#include <sstream>
#include <tuple>

std::tuple
    < std::string
    , std::vector<std::string>
    > zfx_compile
    ( std::string const &code
    ) {
    cout << "=== ZFX" << endl;
    cout << code << endl;

    cout << "=== Parse" << endl;
    auto asts = parse(code);
    for (auto const &a: asts) {
        a->print();
        cout << endl;
    }

    cout << "=== LowerAST" << endl;
    auto
        [ ir
        , symbols
        ] = lower_ast
        ( std::move(asts)
        );
    ir->print();

    cout << "=== LowerMath" << endl;
    ir = apply_lower_math(ir.get());
    ir->print();

    cout << "=== LowerAccess" << endl;
    ir = apply_lower_access(ir.get());
    ir->print();

    cout << "=== EmitAssembly" << endl;
    auto assem = apply_emit_assembly(ir.get());
    cout << assem;

    cout << "=== Assemble" << endl;
    return
        { assem
        , symbols
        };
}

int main() {
    std::string code("pos = pos + 1");
    auto 
        [ assem
        , symbols
        ] = zfx_compile
        ( code
        );

    auto prog = assemble_program(assem);

    float arr[4] = {1, 2, 3, 4};
    prog->set_channel_pointer(0, arr);
    prog->execute();

    printf("result:");
    for (auto val: arr) printf(" %f", val);
    printf("\n");

    return 0;
}
