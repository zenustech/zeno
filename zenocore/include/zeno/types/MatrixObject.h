#pragma once

#include <zeno/core/IObject.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <variant>

namespace zeno
{
    struct MatrixObject
        : zeno::IObjectClone<MatrixObject>
    {
        std::variant<glm::mat3, glm::mat4> m;
    }; // struct MatrixObject

} // namespace zeno