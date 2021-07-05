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
        std::map<std::string, std::string> inityping;
        inityping["@pos"] = "f3";
        inityping["@vel"] = "f3";
        inityping["@clr"] = "f3";
        auto asmcode = compile_program(inityping, code);
        program->prog = assemble_program(asmcode);
        set_output("program", std::move(program));
    }
};

ZENDEFNODE(CompileProgram, {
    {"code"},
    {"program"},
    {},
    {"zenofx"},
});
