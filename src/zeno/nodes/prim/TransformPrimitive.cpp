#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/NumericObject.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/quaternion.hpp>
#include <cstring>

namespace zeno {


struct TransformPrimitive : zeno::INode {
    static glm::vec3 mapplypos(glm::mat4 const &matrix, glm::vec3 const &vector) {
        auto vector4 = matrix * glm::vec4(vector, 1.0f);
        return glm::vec3(vector4) / vector4.w;
    }


    static glm::vec3 mapplynrm(glm::mat4 const &matrix, glm::vec3 const &vector) {
        glm::mat3 normMatrix(matrix);
        normMatrix = glm::transpose(glm::inverse(normMatrix));
        auto vector3 = normMatrix * vector;
        return glm::normalize(vector3);
    }

    virtual void apply() override {
        zeno::vec3f translate = {0,0,0};
        zeno::vec4f rotation = {0,0,0,1};
        zeno::vec3f eulerXYZ = {0,0,0};
        zeno::vec3f scaling = {1,1,1};
        if (has_input("translation"))
            translate = get_input<zeno::NumericObject>("translation")->get<zeno::vec3f>();
        if (has_input("eulerXYZ"))
            eulerXYZ = get_input<zeno::NumericObject>("eulerXYZ")->get<zeno::vec3f>();
        if (has_input("quatRotation"))
            rotation = get_input<zeno::NumericObject>("quatRotation")->get<zeno::vec4f>();
        if (has_input("scaling"))
            scaling = get_input<zeno::NumericObject>("scaling")->get<zeno::vec3f>();
        glm::mat4 matTrans = glm::translate(glm::vec3(translate[0], translate[1], translate[2]));
        glm::mat4 matRotx  = glm::rotate( eulerXYZ[0], glm::vec3(1,0,0) );
        glm::mat4 matRoty  = glm::rotate( eulerXYZ[1], glm::vec3(0,1,0) );
        glm::mat4 matRotz  = glm::rotate( eulerXYZ[2], glm::vec3(0,0,1) );
        glm::quat myQuat(rotation[3], rotation[0], rotation[1], rotation[2]);
        glm::mat4 matQuat  = glm::toMat4(myQuat);
        glm::mat4 matScal  = glm::scale( glm::vec3(scaling[0], scaling[1], scaling[2] ));
        auto matrix = matTrans*matRotz*matRoty*matRotx*matQuat*matScal;

        auto prim = get_input<PrimitiveObject>("prim");
        auto outprim = std::make_unique<PrimitiveObject>(*prim);

        if (prim->has_attr("pos")) {
            auto &pos = outprim->attr<zeno::vec3f>("pos");
            #pragma omp parallel for
            for (int i = 0; i < pos.size(); i++) {
                auto p = zeno::vec_to_other<glm::vec3>(pos[i]);
                p = mapplypos(matrix, p);
                pos[i] = zeno::other_to_vec<3>(p);
            }
        }

        if (prim->has_attr("nrm")) {
            auto &nrm = outprim->attr<zeno::vec3f>("nrm");
            #pragma omp parallel for
            for (int i = 0; i < nrm.size(); i++) {
                auto n = zeno::vec_to_other<glm::vec3>(nrm[i]);
                n = mapplynrm(matrix, n);
                nrm[i] = zeno::other_to_vec<3>(n);
            }
        }
        set_output("outPrim", std::move(outprim));
    }
};

ZENDEFNODE(TransformPrimitive, {
    {
    {"PrimitiveObject", "prim"},
    {"vec3f", "translation", "0,0,0"},
    {"vec3f", "eulerXYZ", "0,0,0"},
    {"vec4f", "quatRotation", "0,0,0,1"},
    {"vec3f", "scaling", "1,1,1"},
    },
    {
    {"PrimitiveObject", "outPrim"}
    },
    {},
    {"primitive"},
});


struct TranslatePrimitive : zeno::INode {
    virtual void apply() override {
        auto translation = get_input<zeno::NumericObject>("translation")->get<zeno::vec3f>();
        auto prim = get_input<PrimitiveObject>("prim");
        auto &pos = prim->attr<vec3f>("pos");
        #pragma omp parallel for
        for (int i = 0; i < pos.size(); i++) {
            pos[i] += translation;
        }
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(TranslatePrimitive, {
    {
    {"PrimitiveObject", "prim"},
    {"vec3f", "translation", "0,0,0"},
    },
    {"prim"},
    {},
    {"primitive"},
});


}
