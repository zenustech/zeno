#include <array>
#include <zeno/zeno.h>
#include <zeno/extra/ShaderNode.h>
#include <zeno/types/ShaderObject.h>

namespace zeno {
    
struct ImplShaderBuffer : ShaderNodeClone<ImplShaderBuffer> {
    int out;

    virtual int determineType(EmissionPass *em) override {
        return TypeHint.at("uint64");
    }

    virtual void emitCode(EmissionPass *em) override {

        auto name = get_input2<std::string>("name");

        if ( out > 0 ) {
            return em->emitCode(name + "_bfsize");
        }
        return em->emitCode("reinterpret_cast<uint64_t>("+ name + "_buffer)");
    }
};

struct ShaderBuffer : INode {

    virtual void apply() override {

        static const auto list = std::array {"out", "size"}; 

        for(int i=0; i<list.size(); ++i) {

            auto node = std::make_shared<ImplShaderBuffer>();
            node->inputs["name"] = get_input("name");
            node->out = i;
            auto shader = std::make_shared<ShaderObject>(node.get());
            set_output(list[i], std::move(shader));
        }
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
        em->determineType(get_input("offset").get());

        auto type = get_input2<std::string>("type");
        return TypeHint.at(type);
    }

    virtual void emitCode(EmissionPass *em) override {

        auto buffer = get_input("buffer").get();

        auto in = em->determineExpr(buffer);
        auto type = get_input2<std::string>("type");
        auto offset = get_input("offset").get();
        auto offsetVar = em->determineExpr(offset);

        em->emitCode("buffer_read<" + type + ">("+ in + "," + offsetVar + ")" );
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