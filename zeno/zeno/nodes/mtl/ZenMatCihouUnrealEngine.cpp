#include <zeno/zeno.h>
#include <zeno/extra/ZenMatNode.h>
#include <zeno/types/ZenMatObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/utils/string.h>

namespace zeno {

struct ZenMatCihouUnrealEngine : INode {
    virtual void apply() override {
        EmissionPass em;
        auto frag = get_input<StringObject>("frag")->get();
        auto commonCode = get_input<StringObject>("common")->get();

        std::string code = R"(
/* BEGIN CIHOU SON OF A MICROSOFT */
)" + commonCode + R"(
FMaterialAttributes zenoUnrealShader(float3 in_pos, float3 in_clr, float3 in_nrm) {
float3 att_pos = in_pos;  /* world space position */
float3 att_clr = in_clr;  /* vertex color */
float3 att_nrm = in_nrm;  /* world space normal */
/* custom_shader_begin */
)" + frag + R"(
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

ZENDEFNODE(ZenMatCihouUnrealEngine, {
    {
        {"string", "frag"},
        {"string", "common"},
    },
    {
        {"string", "code"},
    },
    {
    },
    {"zenMat"},
});

}
