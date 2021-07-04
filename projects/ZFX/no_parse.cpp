#include <zeno/zeno.h>
#include <zeno/StringObject.h>
#include "program.h"
#include "parse.h"
#include <memory>

using namespace zeno;

struct ProgramObject {
    Program prog;
};

struct ParseProgram : INode {
    virtual void apply() override {
        auto code = get_input<StringObject>("code")->get();
        auto program = std::make_shared<ProgramObject>();
        program->prog = parse_program(code);
        set_output("program", std::move(program));
    }
};

ZENDEFNODE(ParseProgram, {
    {"code"},
    {"program"},
    {},
    {"zenofx"},
});
