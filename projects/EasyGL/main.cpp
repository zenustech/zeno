#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <GLES3/gl3.h>

namespace zeno::easygl {

struct GLPrimitiveObject : zeno::IObjectClone<GLPrimitiveObject,
    zeno::PrimitiveObject> {
    std::vector<std::string> boundAttrs;

    void add_bound_attr(std::string const &name) {
        boundAttrs.push_back(name);
    }
};

struct GLShaderObject : zeno::IObjectClone<GLShaderObject> {
    struct Impl {
    };

    std::shared_ptr<Impl> impl;
};

struct GLCreateShader : zeno::INode {
    virtual void apply() override {
        auto source = get_input<GLPrimitiveObject>("source");
    }
};

ZENDEFNODE(GLCreateShader, {
        {"source"},
        {},
        {},
        {"EasyGL"},
});

struct GLDrawArrayTriangles : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<GLPrimitiveObject>("prim");
        for (int i = 0; i < prim->boundAttrs.size(); i++) {
            auto name = prim->boundAttrs[i];
            std::visit([] (auto const &arr) {
                using T = std::decay_t<decltype(arr[0])>;
                using S = zeno::decay_vec_t<T>;
                glVertexAttribPointer
                        ( /*index=*/0
                        , /*size=*/zeno::is_vec_n<T>
                        , GL_FLOAT
                        , GL_FALSE
                        , /*stride=*/0
                        , (void *)arr.data()
                        );
            }, prim->attr(name));
        }
        glDrawArrays(GL_TRIANGLES, 0, /*count=*/prim->size());
    }
};

ZENDEFNODE(GLDrawArrayTriangles, {
        {"prim"},
        {},
        {},
        {"EasyGL"},
});

struct MakeSimpleTriangle : zeno::INode {
    inline static const GLfloat vVertices[] = {
        0.0f,  0.5f, 0.0f,
        -0.5f, -0.5f, 0.0f,
        0.5f, -0.5f, 0.0f,
    };

    virtual void apply() override {
        auto prim = std::make_shared<GLPrimitiveObject>();
        prim->add_attr<zeno::vec3f>("pos");
        prim->add_bound_attr("pos");
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(MakeSimpleTriangle, {
        {},
        {"prim"},
        {},
        {"EasyGL"},
});

}
