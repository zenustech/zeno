#include "LowerAST.h"
#include "Visitors.h"
#include "x64/Program.h"

int main() {
    std::string code("pos = pos + 1");
    cout << "==============" << endl;
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
    auto inst = assemble_program(assem);

    cout << "==============" << endl;
    return 0;
}
