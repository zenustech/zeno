#include "AST.h"
#include "IR.h"
#include "Visitors.h"

int main() {
    std::string code("pos = 1 + (2 + x*4) * 3");
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
    auto new_ir = apply_clone(ir.get());
    ir = nullptr;
    new_ir->print();

    cout << "==============" << endl;
    return 0;
}
