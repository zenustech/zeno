#include <zeno/zeno.h>
#include <zeno/StringObject.h>
#include "no_ProgramObject.h"
#include "program.h"
#include "compile.h"
#include <memory>

using namespace zeno;

struct CompileProgram : INode {
    virtual void apply() override {
        auto code = get_input<StringObject>("code")->get();
        auto program = std::make_shared<ProgramObject>();
        program->prog = assemble_program(code);
        set_output("program", std::move(program));
    }
};

ZENDEFNODE(CompileProgram, {
    {"code"},
    {"program"},
    {},
    {"zenofx"},
});
