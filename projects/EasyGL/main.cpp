#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <GLES3/gl3.h>

namespace zeno::easygl {

struct GLPrimitiveObject : zeno::IObjectClone<GLPrimitiveObject,
    zeno::PrimitiveObject> {
    std::vector<std::string> boundAttrs;
};

struct GLDrawPoints : zeno::INode {
    inline static const GLfloat vVertices[] = {
        0.0f,  0.5f, 0.0f,
        -0.5f, -0.5f, 0.0f,
        0.5f, -0.5f, 0.0f,
    };

    virtual void apply() override {
        auto prim = get_input<GLPrimitiveObject>("prim");
        for (auto const &attrName: prim->boundAttrs) {
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, vVertices);
        }
        glDrawArrays(GL_TRIANGLES, 0, 3);
    }
};

}
