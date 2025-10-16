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

        auto obj = ZImpl(get_input_shader("in"));
        em->determineType(obj);

        auto type = zsString2Std(get_input2_string("type"));
        return TypeHint.at(type);
    }

    virtual void emitCode(EmissionPass *em) override {

        auto op = zsString2Std(get_input2_string("op"));
        auto type = zsString2Std(get_input2_string("type"));
        
        auto obj = ZImpl(get_input_shader("in"));
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
        {gParamType_Int, "in"}
    },
    {
        {gParamType_Shader, "out"},
    },
    {
        {"enum bit_cast data_cast ", "op", "bit_cast"},
        {"enum " + ShaderDataTypeNamesString, "type", "bool"},
    },
    {"shader"},
});

struct ShaderPrint : ShaderNodeClone<ShaderPrint> {
    virtual int determineType(EmissionPass *em) override {
        auto in = ZImpl(get_input_shader("in"));
        auto in_type = em->determineType(in);
        return in_type;
    }

    virtual void emitCode(EmissionPass *em) override {
        
        auto in = ZImpl(get_input_shader("in"));
        auto in_type = em->determineType(in);
        auto in_expr = em->determineExpr(in);

        auto str = zsString2Std(get_input2_string("str"));

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
        {gParamType_IObject, "in"},
        {gParamType_String, "str", ""}
    },
    {
        {gParamType_Shader, "out"},
    },
    {},
    {"shader"},
});

}