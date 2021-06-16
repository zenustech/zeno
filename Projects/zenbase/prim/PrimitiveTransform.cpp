#include <zen/zen.h>
#include <zen/PrimitiveObject.h>
#include <zen/NumericObject.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/quaternion.hpp>
#include <cstring>

namespace zen {

static glm::vec3 mapplypos(glm::mat4 const &matrix, glm::vec3 const &vector) {
  auto vector4 = matrix * glm::vec4(vector, 1.0f);
  
  return glm::vec3(vector4) / vector4.w;
}


static glm::vec3 mapplydir(glm::mat4 const &matrix, glm::vec3 const &vector) {
  glm::mat3 normMatrix(matrix);
  normMatrix = glm::transpose(glm::inverse(normMatrix));
  auto vector3 = normMatrix * vector;
  return vector3;
}

struct TransformPrimitive : zen::INode {
    virtual void apply() override {
        zen::vec3f translate = {0,0,0};
        zen::vec4f rotation = {0,0,0,1};
        zen::vec3f eulerXYZ = {0,0,0};
        zen::vec3f scaling = {1,1,1};
        if (has_input("translation"))
            translate = get_input<zen::NumericObject>("translation")->get<zen::vec3f>();
        if (has_input("eulerXYZ"))
            eulerXYZ = get_input<zen::NumericObject>("eulerXYZ")->get<zen::vec3f>();
        if (has_input("quatRotation"))
            rotation = get_input<zen::NumericObject>("quatRotation")->get<zen::vec4f>();
        if (has_input("scaling"))
            scaling = get_input<zen::NumericObject>("scaling")->get<zen::vec3f>();
        glm::mat4 matTrans = glm::translate(glm::vec3(translate[0], translate[1], translate[2]));
        glm::mat4 matRotx  = glm::rotate( eulerXYZ[0], glm::vec3(1,0,0) );
        glm::mat4 matRoty  = glm::rotate( eulerXYZ[1], glm::vec3(0,1,0) );
        glm::mat4 matRotz  = glm::rotate( eulerXYZ[2], glm::vec3(0,0,1) );
        glm::quat myQuat(rotation[0], rotation[1], rotation[2], rotation[3]);
        glm::mat4 matQuat  = glm::toMat4(myQuat);
        glm::mat4 matScal  = glm::scale( glm::vec3(scaling[0], scaling[1], scaling[2] ));
        auto matrix = matTrans*matRotz*matRoty*matRotx*matQuat*matScal;

        auto prim = get_input("prim")->as<PrimitiveObject>();
        auto outprim = zen::IObject::make<PrimitiveObject>(*prim);

        if (prim->has_attr("pos")) {
            auto &pos = prim->attr<zen::vec3f>("pos");
            //#pragma omp parallel for
            for (int i = 0; i < pos.size(); i++) {
                pos[i] = mapplypos(matrix, pos[i]);
            }
        }

        if (prim->has_attr("nrm")) {
            auto &nrm = prim->attr<zen::vec3f>("nrm");
            //#pragma omp parallel for
            for (int i = 0; i < nrm.size(); i++) {  // TODO: inverse-transpose??
                nrm[i] = glm::normalize(mapplydir(matrix, nrm[i]));
            }
        }
        set_output("outPrim", outprim);
    }
};

ZENDEFNODE(PrimitiveTransform, {
    {"prim", "translation", "eulerXYZ", "quatRotation", "scaling"},
    {"outPrim"},
    {},
    {"primitive"},
}
