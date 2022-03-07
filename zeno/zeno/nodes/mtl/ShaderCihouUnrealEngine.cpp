#include <zeno/zeno.h>
#include <zeno/extra/ShaderNode.h>
#include <zeno/types/MaterialObject.h>
#include <zeno/types/ShaderObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/utils/string.h>

namespace zeno {

struct ShaderCihouUnrealEngine : INode {
    virtual void apply() override {
        EmissionPass em;
        auto mtl = get_input<MaterialObject>("mtl");

        std::string code = R"(
/* BEGIN CIHOU SON OF A MICROSOFT */
)" + mtl->common + R"(
FMaterialAttributes zenoUnrealShader(float3 in_pos, float3 in_clr, float3 in_nrm) {
float3 att_pos = in_pos;  /* world space position */
float3 att_clr = in_clr;  /* vertex color */
float3 att_nrm = in_nrm;  /* world space normal */
/* custom_shader_begin */
)" + mtl->frag + R"(
/* custom_shader_end */
FMaterialAttributes mat = (FMaterialAttributes)0;
mat.BaseColor = mat_basecolor;
mat.Metallic = mat_metallic;
mat.Specular = mat_specular;
mat.Roughness = mat_roughness;
mat.Normal = mat_normal;
mat.EmissiveColor = mat_emission;
return mat;
}
/* END CIHOU SON OF A MICROSOFT */
)";

        set_output2("code", std::move(code));
    }
};

ZENDEFNODE(ShaderCihouUnrealEngine, {
    {
        {"MaterialObject", "mtl"},
    },
    {
        {"string", "code"},
    },
    {
    },
    {"shader"},
});

}
