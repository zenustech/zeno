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
        for (int i = 0; i < prim->boundAttrs.size(); i++) {
            auto name = prim->boundAttrs[i];
            std::visit([] (auto const &arr) {
                using T = std::decay_t<decltype(arr[0])>;
                using S = zeno::decay_vec_t<T>;
                if constexpr (zeno::is_vec_v<T>) {
                    glVertexAttribPointer
                            ( /*index=*/0
                            , /*size=*/zeno::is_vec_n<T>
                            , GL_FLOAT
                            , GL_FALSE
                            , /*stride=*/0
                            , vVertices
                            );
                }
            }, prim->attr(name));
        }
        glDrawArrays(GL_TRIANGLES, 0, /*count=*/3);
    }
};

}
