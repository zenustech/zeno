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

    int maxregs = 0;

    void define(int dst) {
        maxregs = std::max(maxregs, dst + 1);
    }

    void addAssign(int dst, int src) {
        define(dst);
        oss << "    r" << dst << " = r" << src << ";\n";
    }

    void addLoadLiterial(int dst, std::string const &expr) {
        define(dst);
        oss << "    r" << dst << " = " << expr << ";\n";
    }

    void addLoadArray(int val, std::string const &mem, int id) {
        define(val);
        oss << "    r" << val << " = " << mem << "[" << id << "];\n";
    }

    void addStoreArray(int val, std::string const &mem, int id) {
        oss << "    " << mem << "[" << id << "] = r" << val << ";\n";
    }

    void addMath(const char *name, int dst, int lhs, int rhs) {
        define(dst);
        oss << "    r" << dst << " = r" << lhs
            << " " << name << " r" << rhs << ";\n";
    }

    void addMathFunc(const char *name, int dst, int src) {
        define(dst);
        oss << "    r" << dst << " = " << name << "(r" << src << ");\n";
    }

    void addMathFunc(const char *name, int dst, int lhs, int rhs) {
        define(dst);
        oss << "    r" << dst << " = " << name << "(r" << lhs
            << ", r" << rhs << ");\n";
    }

    void addIf(int cond) {
        oss << "    if (r" << cond << ") {\n";
    }

    void addElseIf(int cond) {
        oss << "    } else if (r" << cond << ") {\n";
    }

    void addElse() {
        oss << "    } else {\n";
    }

    void addEndIf() {
        oss << "    }\n";
    }

    std::string finish(int nlocals) {
        oss_head << "__device__ void zfx_wrangle_func";
        oss_head << "(float *globals, float const *params) {\n";
        oss << "}\n";
        if (nlocals) {
            oss_head << "    float locals[" << nlocals << "];";
        }
        if (maxregs) {
            oss_head << "    float r0";
            for (int i = 1; i < maxregs; i++) {
                oss_head << ", r" << i;
            }
            oss_head << ";\n";
        }
        return oss_head.str() + oss.str();
    }
};

struct ImplAssembler {
    std::unique_ptr<CUDABuilder> builder = std::make_unique<CUDABuilder>();
    std::string code;

    int nparams = 0;
    int nlocals = 0;
    int nglobals = 0;

    static inline std::set<std::string> unary_maths =
        { "sqrt"
        , "sin"
        , "cos"
        , "tan"
        , "asin"
        , "acos"
        , "atan"
        , "exp"
        , "log"
        , "floor"
        , "round"
        , "ceil"
        , "abs"
        , "rsqrt"
        };

    static inline std::set<std::string> binary_maths =
        { "min"
        , "max"
        , "pow"
        , "atan2"
        , "mod"
        };

    static auto wrapup_function(std::string name) {
        name += 'f';
        if (contains({"min", "max", "abs", "mod"}, name)) {
            name = 'f' + name;
        }
        return name;
    }

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
                auto expr = linesep[2];
                builder->addLoadLiterial(dst, expr);

            } else if (cmd == "ldp") {
                ERROR_IF(linesep.size() < 2);
                auto dst = from_string<int>(linesep[1]);
                auto id = from_string<int>(linesep[2]);
                nparams = std::max(nparams, id + 1);
                builder->addLoadArray(dst, "params", id);

            } else if (cmd == "ldl") {
                ERROR_IF(linesep.size() < 2);
                auto dst = from_string<int>(linesep[1]);
                auto id = from_string<int>(linesep[2]);
                nlocals = std::max(nlocals, id + 1);
                builder->addLoadArray(dst, "locals", id);

            } else if (cmd == "stl") {
                ERROR_IF(linesep.size() < 2);
                auto dst = from_string<int>(linesep[1]);
                auto id = from_string<int>(linesep[2]);
                nlocals = std::max(nlocals, id + 1);
                builder->addStoreArray(dst, "locals", id);

            } else if (cmd == "ldg") {
                ERROR_IF(linesep.size() < 2);
                auto dst = from_string<int>(linesep[1]);
                auto id = from_string<int>(linesep[2]);
                nglobals = std::max(nglobals, id + 1);
                builder->addLoadArray(dst, "globals", id);

            } else if (cmd == "stg") {
                ERROR_IF(linesep.size() < 2);
                auto dst = from_string<int>(linesep[1]);
                auto id = from_string<int>(linesep[2]);
                nglobals = std::max(nglobals, id + 1);
                builder->addStoreArray(dst, "globals", id);

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

            } else if (contains(unary_maths, cmd)) {
                ERROR_IF(linesep.size() < 2);
                auto dst = from_string<int>(linesep[1]);
                auto src = from_string<int>(linesep[2]);
                auto op = wrapup_function(cmd);
                builder->addMathFunc(op.c_str(), dst, src);

            } else if (contains(binary_maths, cmd)) {
                ERROR_IF(linesep.size() < 3);
                auto dst = from_string<int>(linesep[1]);
                auto lhs = from_string<int>(linesep[2]);
                auto rhs = from_string<int>(linesep[3]);
                auto op = wrapup_function(cmd);
                builder->addMathFunc(op.c_str(), dst, lhs, rhs);


            } else if (cmd == "mov") {
                ERROR_IF(linesep.size() < 2);
                auto dst = from_string<int>(linesep[1]);
                auto src = from_string<int>(linesep[2]);
                builder->addAssign(dst, src);

            } else if (cmd == ".if") {
                ERROR_IF(linesep.size() < 1);
                auto cond = from_string<int>(linesep[1]);
                builder->addIf(cond);

            } else if (cmd == ".elseif") {
                ERROR_IF(linesep.size() < 1);
                auto cond = from_string<int>(linesep[1]);
                builder->addElseIf(cond);

            } else if (cmd == ".else") {
                builder->addElse();

            } else if (cmd == ".endif") {
                builder->addEndIf();

            } else {
                error("bad assembly command `%s`", cmd.c_str());
            }
        }

        code = builder->finish(nlocals);

#ifdef ZFX_PRINT_IR
        log_printf("params: %d\n", nparams);
        log_printf("locals: %d\n", nlocals);
        log_printf("globals: %d\n", nglobals);
        log_printf("cuda code:\n%s\n", code.c_str());
#endif
    }
};

std::string Assembler::impl_assemble
    ( std::string const &lines
    ) {
    ImplAssembler a;
    a.parse(lines);
    return a.code;
}

}
