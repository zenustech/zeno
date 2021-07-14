#include <zfx/utils.h>
#include <zfx/cuda.h>
#include <sstream>
#include <map>

namespace zfx::cuda {

#define ERROR_IF(x) do { \
    if (x) { \
        error("`%s`", #x); \
    } \
} while (0)

struct CUDABuilder {
    std::ostringstream oss;
    std::ostringstream oss_head;
    float consts[1024];

    int maxregs = 0;

    void define(int dst) {
        maxregs = std::max(maxregs, dst);
    }

    void addAssign(int dst, int src) {
        define(dst);
        oss << "r" << dst << " = r" << src << ";\n";
    }

    void addLoadConst(int dst, int id) {
        define(dst);
        oss << "r" << dst << " = " << consts[id] << ";\n";
    }

    void addLoadLocal(int dst, int id) {
        define(dst);
        oss << "r" << dst << " = locals[" << id << "];\n";
    }

    void addStoreLocal(int dst, int id) {
        oss << "locals[" << id << "] = r" << dst << ";\n";
    }

    void addMath(const char *name, int dst, int lhs, int rhs) {
        define(dst);
        oss << "r" << dst << " = r" << lhs
            << " " << name << " r" << rhs << ";\n";
    }

    void addMathFunc(const char *name, int dst, int src) {
        define(dst);
        oss << "r" << dst << " = " << name << "(r" << src << ");\n";
    }

    std::string finish() {
        oss_head << "__device__ void zfx_wrangle_func";
        oss_head << "(float *locals, float *params) {\n";
        oss << "}\n";
        if (maxregs) {
            oss_head << "float r0";
            for (int i = 1; i < maxregs; i++) {
                oss_head << ", r" << i;
            }
            oss_head << ";\n";
        }
        return oss_head.str() + oss.str();
    }
};

struct Assembler {
    std::unique_ptr<CUDABuilder> builder = std::make_unique<CUDABuilder>();
    std::unique_ptr<Program> prog = std::make_unique<Program>();

    int nconsts = 0;
    int nlocals = 0;
    //int nglobals = 0;

    void parse(std::string const &lines) {
        for (auto line: split_str(lines, '\n')) {
            if (!line.size()) continue;

            auto linesep = split_str(line, ' ');
            ERROR_IF(linesep.size() < 1);
            auto cmd = linesep[0];
            if (0) {

            } else if (cmd == "ldi") {
                ERROR_IF(linesep.size() < 2);
                auto dst = from_string<int>(linesep[1]);
                auto id = from_string<int>(linesep[2]);
                nconsts = std::max(nconsts, id + 1);
                builder->addLoadConst(dst, id);

            } else if (cmd == "ldm") {
                ERROR_IF(linesep.size() < 2);
                auto dst = from_string<int>(linesep[1]);
                auto id = from_string<int>(linesep[2]);
                nlocals = std::max(nlocals, id + 1);
                builder->addLoadLocal(dst, id);

            } else if (cmd == "stm") {
                ERROR_IF(linesep.size() < 2);
                auto dst = from_string<int>(linesep[1]);
                auto id = from_string<int>(linesep[2]);
                nlocals = std::max(nlocals, id + 1);
                builder->addStoreLocal(dst, id);

            } else if (cmd == "add") {
                ERROR_IF(linesep.size() < 3);
                auto dst = from_string<int>(linesep[1]);
                auto lhs = from_string<int>(linesep[2]);
                auto rhs = from_string<int>(linesep[3]);
                builder->addMath("+", dst, lhs, rhs);

            } else if (cmd == "sub") {
                ERROR_IF(linesep.size() < 3);
                auto dst = from_string<int>(linesep[1]);
                auto lhs = from_string<int>(linesep[2]);
                auto rhs = from_string<int>(linesep[3]);
                builder->addMath("-", dst, lhs, rhs);

            } else if (cmd == "mul") {
                ERROR_IF(linesep.size() < 3);
                auto dst = from_string<int>(linesep[1]);
                auto lhs = from_string<int>(linesep[2]);
                auto rhs = from_string<int>(linesep[3]);
                builder->addMath("*", dst, lhs, rhs);

            } else if (cmd == "div") {
                ERROR_IF(linesep.size() < 3);
                auto dst = from_string<int>(linesep[1]);
                auto lhs = from_string<int>(linesep[2]);
                auto rhs = from_string<int>(linesep[3]);
                builder->addMath("/", dst, lhs, rhs);

            } else if (cmd == "sqrt") {
                ERROR_IF(linesep.size() < 2);
                auto dst = from_string<int>(linesep[1]);
                auto src = from_string<int>(linesep[2]);
                builder->addMathFunc("sqrt", dst, src);

            } else if (cmd == "mov") {
                ERROR_IF(linesep.size() < 2);
                auto dst = from_string<int>(linesep[1]);
                auto src = from_string<int>(linesep[2]);
                builder->addAssign(dst, src);

            } else {
                error("bad assembly command `%s`", cmd.c_str());
            }
        }

        auto code = builder->finish();

#ifdef ZFX_PRINT_IR
        printf("cuda code:\n");
        printf("%s", code.c_str());
        printf("\n");
#endif

        prog->code = code;
    }

    void set_constants(std::map<int, std::string> const &consts) {
        for (auto const &[idx, expr]: consts) {
            if (!(std::istringstream(expr) >> builder->consts[idx])) {
                error("cannot parse literial constant `%s`",
                    expr.c_str());
            }
        }
    }
};

std::unique_ptr<Program> Program::assemble
    ( std::string const &lines
    , std::map<int, std::string> const &consts
    ) {
    Assembler a;
    a.set_constants(consts);
    a.parse(lines);
    return std::move(a.prog);
}

}
