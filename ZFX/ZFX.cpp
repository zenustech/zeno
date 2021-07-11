#include "LowerAST.h"
#include "Visitors.h"
#include "x64/Program.h"
#include <sstream>
#include <tuple>

namespace zfx {

std::tuple
    < std::string
    , std::vector<std::string>
    > zfx_compile
    ( std::string const &code
    ) {
#ifdef ZFX_PRINT_IR
    cout << "=== ZFX" << endl;
    cout << code << endl;
#endif

#ifdef ZFX_PRINT_IR
    cout << "=== Parse" << endl;
#endif
    auto asts = parse(code);
#ifdef ZFX_PRINT_IR
    for (auto const &a: asts) {
        a->print();
        cout << endl;
    }
#endif

#ifdef ZFX_PRINT_IR
    cout << "=== LowerAST" << endl;
#endif
    auto
        [ ir
        , symbols
        ] = lower_ast
        ( std::move(asts)
        );
#ifdef ZFX_PRINT_IR
    ir->print();
#endif

#ifdef ZFX_PRINT_IR
    cout << "=== LowerMath" << endl;
#endif
    ir = apply_lower_math(ir.get());
#ifdef ZFX_PRINT_IR
    ir->print();
#endif

#ifdef ZFX_PRINT_IR
    cout << "=== LowerAccess" << endl;
#endif
    ir = apply_lower_access(ir.get());
#ifdef ZFX_PRINT_IR
    ir->print();
#endif

#ifdef ZFX_PRINT_IR
    cout << "=== EmitAssembly" << endl;
#endif
    auto assem = apply_emit_assembly(ir.get());
#ifdef ZFX_PRINT_IR
    cout << assem;
#endif

#ifdef ZFX_PRINT_IR
    cout << "=== Assemble" << endl;
#endif
    return
        { assem
        , symbols
        };
}

}

int main() {
    using namespace zfx;

    std::string code("pos = pos + 0.5");
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
