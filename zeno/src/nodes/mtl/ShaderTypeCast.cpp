#include <zeno/zeno.h>
#include <zeno/extra/ShaderNode.h>
#include <zeno/types/ShaderObject.h>

#include <iomanip>
#include <sstream>
#include <iostream>

namespace zeno {

static std::string dataTypeDefaultString() {
    return ShaderDataTypeNames.front();
}

struct ShaderTypeCast : ShaderNodeClone<ShaderTypeCast> {
    virtual int determineType(EmissionPass *em) override {

        auto obj = get_input("in").get();
        em->determineType(obj);

        auto type = get_input2<std::string>("type:");
        return TypeHint.at(type);
    }

    virtual void emitCode(EmissionPass *em) override {

        auto op = get_input2<std::string>("op:");
        auto type = get_input2<std::string>("type:");
        
        auto obj = get_input("in").get();
        auto in = em->determineExpr(obj);
        
        if (op == "bit_cast") {
            em->emitCode("reinterpret_cast<"+type+"&>("+in+")");
        } else {
            em->emitCode(type + "(" + in + ")" );
        }
    }
};

ZENDEFNODE(ShaderTypeCast, {
    {
        {"in"}
    },
    {
        {"shader", "out"},
    },
    {
        {"enum bit_cast data_cast ", "op", "bit_cast"},
        {"enum " + ShaderDataTypeNamesString, "type", "bool"},
    },
    {"shader"},
});

struct ShaderPrint : ShaderNodeClone<ShaderPrint> {
    virtual int determineType(EmissionPass *em) override {
        auto in = get_input("in").get();
        auto in_type = em->determineType(in);
        return in_type;
    }

    virtual void emitCode(EmissionPass *em) override {
        
        auto in = get_input("in").get();
        auto in_type = em->determineType(in);
        auto in_expr = em->determineExpr(in);

        auto str = get_input2<std::string>("str");

        static const std::map<std::string, std::string> typeTable {
            {"float", "%f"}, {"bool", "%d"},
            {"int", "%d"}, {"uint", "%u"},
            {"int64", "%ll"}, {"uint64", "%llu"}
        };

        auto typeStr = TypeHintReverse.at(in_type);
        auto typePri = typeTable.at(typeStr);
        auto content = str + ": " + typePri + "\\n";

        std::stringstream ss;
        ss << "printf(";
        ss << std::quoted(content, '"', '\n');
        ss << "," << in_expr << ");";

        em->lines.back() += ss.str();
        em->emitCode(in_expr);
    }
};

ZENDEFNODE(ShaderPrint, {
    {
        {"in"},
        {"string", "str", ""}
    },
    {
        {"shader", "out"},
    },
    {},
    {"shader"},
});

}