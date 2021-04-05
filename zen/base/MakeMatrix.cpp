#include <zen/zen.h>
#include <zen/MatrixObject.h>
#include <glm/gtc/matrix_transform.hpp>
#include <cstring>

namespace zenbase {

struct MakeMatrix : zen::INode {
  virtual void apply() override {
    auto position = zen::get_float3<glm::vec3>(get_param("position"));
    auto rotation = zen::get_float3<glm::vec3>(get_param("rotation"));
    auto scale = zen::get_float3<glm::vec3>(get_param("scale"));
    auto matrix = zen::IObject::make<MatrixObject>();
    matrix->m = glm::translate(
        glm::mat4(1),
        position);
    set_output("matrix", matrix);
  }
};

static int defMakeMatrix = zen::defNodeClass<MakeMatrix>("MakeMatrix",
    { /* inputs: */ {
    }, /* outputs: */ {
    "matrix",
    }, /* params: */ {
    {"float3","position","0 0 0"},
    {"float3","rotation","0 0 0"},
    {"float3","scale","0 0 0"},
    }, /* category: */ {
    "misc",
    }});

}
