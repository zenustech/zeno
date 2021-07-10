#include "AST.h"
#include "IR.h"
#include "Visitors.h"

int main() {
    std::string code("pos = 1");
    cout << code << endl;

    cout << "==============" << endl;
    auto asts = parse(code);
    for (auto const &a: asts) {
        a->print();
        cout << endl;
    }

    cout << "==============" << endl;
    auto ir = lower_ast(std::move(asts));
    ir->print();

    cout << "==============" << endl;
    ir = apply_lower_access(ir.get());
    ir->print();

    cout << "==============" << endl;
    auto assem = apply_emit_assembly(ir.get());
    cout << assem;

    cout << "==============" << endl;
    return 0;
}
