#include <glm/ext/matrix_transform.hpp>
#include <type_traits>
#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/NumericObject.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/quaternion.hpp>
#include <cstring>
#include <variant>
namespace zeno {
struct MatrixObject : zeno::IObjectClone<MatrixObject>{//ZhxxHappyObject
    std::variant<glm::mat3, glm::mat4> m;
};

/*struct SetMatrix : zeno::INode{//ZHXX: use Assign instead!fixed using iobjclone now
    virtual void apply() override {
        auto &dst = std::get<glm::mat4>(get_input<zeno::MatrixObject>("dst")->m);
        auto &src = std::get<glm::mat4>(get_input<zeno::MatrixObject>("src")->m);
        dst = src;
    }
};
ZENDEFNODE(SetMatrix, {
    {
    {"dst" },
    {"src" },
    },
    {},
    {},
    {"math"},
});*/

struct MakeLocalSys : zeno::INode{
    virtual void apply() override {
        zeno::vec3f front = {1,0,0};
        zeno::vec3f up = {0,1,0};
        zeno::vec3f right = {0,0,1};
        if (has_input("front"))
            front = get_input<zeno::NumericObject>("front")->get<zeno::vec3f>();
        if (has_input("up"))
            up = get_input<zeno::NumericObject>("up")->get<zeno::vec3f>();
        if (has_input("right"))
            right = get_input<zeno::NumericObject>("right")->get<zeno::vec3f>();

        auto oMat = std::make_shared<MatrixObject>();
        oMat->m = glm::mat4(glm::mat3(front[0], up[0], right[0],
                            front[1], up[1], right[1],
                            front[2], up[2], right[2]));
        set_output("LocalSys", oMat);                    
    }
};
ZENDEFNODE(MakeLocalSys, {
    {
    {"vec3f", "front", "1,0,0"},
    {"vec3f", "up", "0,1,0"},
    {"vec3f", "right", "0,0,1"},
    },
    {{"LocalSys"}},
    {},
    {"math"},
});

struct TransformPrimitive : zeno::INode {//TODO: refactor with boolean variant
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
        zeno::vec3f offset = {0,0,0};
        glm::mat4 pre_mat = glm::mat4(1.0);
        glm::mat4 local = glm::mat4(1.0);
        if (has_input("Matrix"))
            pre_mat = std::get<glm::mat4>(get_input<zeno::MatrixObject>("Matrix")->m);
        if (has_input("translation"))
            translate = get_input<zeno::NumericObject>("translation")->get<zeno::vec3f>();
        if (has_input("eulerXYZ"))
            eulerXYZ = get_input<zeno::NumericObject>("eulerXYZ")->get<zeno::vec3f>();
        if (has_input("quatRotation"))
            rotation = get_input<zeno::NumericObject>("quatRotation")->get<zeno::vec4f>();
        if (has_input("scaling"))
            scaling = get_input<zeno::NumericObject>("scaling")->get<zeno::vec3f>();
        if (has_input("offset"))
            offset = get_input<zeno::NumericObject>("offset")->get<zeno::vec3f>();
        if (has_input("local"))
           local = std::get<glm::mat4>(get_input<zeno::MatrixObject>("local")->m);


        glm::mat4 matTrans = glm::translate(glm::vec3(translate[0], translate[1], translate[2]));
        glm::mat4 matRotx  = glm::rotate( eulerXYZ[0], glm::vec3(1,0,0) );
        glm::mat4 matRoty  = glm::rotate( eulerXYZ[1], glm::vec3(0,1,0) );
        glm::mat4 matRotz  = glm::rotate( eulerXYZ[2], glm::vec3(0,0,1) );
        glm::quat myQuat(rotation[3], rotation[0], rotation[1], rotation[2]);
        glm::mat4 matQuat  = glm::toMat4(myQuat);
        glm::mat4 matScal  = glm::scale( glm::vec3(scaling[0], scaling[1], scaling[2] ));
        auto matrix = pre_mat*local*matTrans*matRotz*matRoty*matRotx*matQuat*matScal*glm::translate(glm::vec3(offset[0], offset[1], offset[2]))*glm::inverse(local);

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
        auto oMat = std::make_shared<MatrixObject>();
        oMat->m = matrix;
        set_output("outPrim", std::move(outprim));
        set_output("Matrix", oMat);
    }
};

ZENDEFNODE(TransformPrimitive, {
    {
    {"PrimitiveObject", "prim"},
    {"vec3f", "translation", "0,0,0"},
    {"vec3f", "offset", "0,0,0"},
    {"vec3f", "eulerXYZ", "0,0,0"},
    {"vec4f", "quatRotation", "0,0,0,1"},
    {"vec3f", "scaling", "1,1,1"},
    {"Matrix"},
    {"local"},
    },
    {{"outPrim"}, {"Matrix"}},
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
