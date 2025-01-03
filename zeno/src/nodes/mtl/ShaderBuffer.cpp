#include <zeno/zeno.h>
#include <zeno/extra/ShaderNode.h>
#include <zeno/types/ShaderObject.h>

namespace zeno {

struct ShaderBuffer : ShaderNodeClone<ShaderBuffer> {
    virtual int determineType(EmissionPass *em) override {
        return TypeHint.at("uint64");
    }

    virtual void emitCode(EmissionPass *em) override {

        auto name = get_input2<std::string>("name");
        return em->emitCode("reinterpret_cast<uint64_t>("+ name + "_buffer" +")");
    }
};

ZENDEFNODE(ShaderBuffer, {
                {
                    {"string", "name", ""},
                },
                {
                    {"shader", "out"},
                    {"int",   "size"},
                },
                {},
                {"shader"},
            });

struct ShaderBufferRead : ShaderNodeClone<ShaderBufferRead> {
    
    virtual int determineType(EmissionPass *em) override {

        em->determineType(get_input("buffer").get());

        auto type = get_input2<std::string>("type");
        return TypeHint.at(type);
    }

    virtual void emitCode(EmissionPass *em) override {

        auto buffer = get_input("buffer").get();

        auto in = em->determineExpr(buffer);
        auto type = get_input2<std::string>("type");
        auto offset = get_input2<int>("offset");

        em->emitCode("buffer_read<" + type + ">("+ in + "," + std::to_string(offset) + ")" );
    }
};

ZENDEFNODE(ShaderBufferRead, {
                {
                    {"buffer"},
                    {"int", "offset", "0"},
                    {"enum " + ShaderDataTypeNamesString, "type", "float"},
                },
                {
                    {"out"},
                },
                {},
                {"shader"},
            });

}