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

    auto ir = lower_ast(asts);
    ir->print();

    DemoVisitor demo;
    demo.apply(ir.get());

    cout << "==============" << endl;
    return 0;
}
