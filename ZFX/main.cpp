#include "LowerAST.h"
#include "Parser.h"
#include "Tokenizer.h"
#include "Visitors.h"

int main() {
    std::string code("pos = 1 + (2 + x*4) * 3");
    cout << code << endl;

    cout << "==============" << endl;
    auto tokens = tokenize(code.c_str());
    for (auto const &t: tokens) {
        cout << t << ' ';
    }
    cout << endl;

    cout << "==============" << endl;
    Parser parser(tokens);
    auto asts = parser.parse();
    for (auto const &a: asts) {
        a->print();
        cout << endl;
    }

    LowerAST lower;
    for (auto const &a: asts) {
        lower.serialize(a.get());
    }
    auto ir = std::move(lower.ir);
    ir->print();

    DemoVisitor demo;
    demo.do_visit(ir.get());

    cout << "==============" << endl;
    return 0;
}
