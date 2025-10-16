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

        auto name = ZImpl(get_input2<std::string>("name"));

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

            auto node = std::make_unique<ImplShaderBuffer>();

            std::string na = zsString2Std(get_input2_string("name"));
            node->m_pAdapter->set_primitive_input("name", na);

            node->out = i;
            auto shader = std::make_unique<ShaderObject>(node.get());
            set_output(list[i], std::move(shader));
        }
    }
};

ZENDEFNODE(ShaderBuffer, {
                {
                    {gParamType_String, "name", ""},
                },
                {
                    {gParamType_Shader, "out"},
                    {gParamType_Int,   "size"},
                },
                {},
                {"shader"},
            });

struct ShaderBufferRead : ShaderNodeClone<ShaderBufferRead> {
    
    virtual int determineType(EmissionPass *em) override {

        em->determineType(ZImpl(get_input_shader("buffer")));

        auto type = ZImpl(get_input2<std::string>("type"));
        return TypeHint.at(type);
    }

    virtual void emitCode(EmissionPass *em) override {

        auto buffer = ZImpl(get_input_shader("buffer"));

        auto in = em->determineExpr(buffer);
        auto type = ZImpl(get_input2<std::string>("type"));
        auto offset = ZImpl(get_input2<int>("offset"));

        em->emitCode("buffer_read<" + type + ">("+ in + "," + std::to_string(offset) + ")" );
    }
};

ZENDEFNODE(ShaderBufferRead, {
                {
                    {gParamType_Shader, "buffer"},
                    {gParamType_Int, "offset", "0"},
                    {"enum " + ShaderDataTypeNamesString, "type", "float"},
                },
                {
                    {gParamType_Shader, "out"},
                },
                {},
                {"shader"},
            });

}