#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/ListObject.h>
#include <GLES2/gl2.h>
#include <cstring>
#include "GLPrimitiveObject.h"

struct GLCreateShader : zeno::INode {
    virtual void apply() override {
        auto typeStr = get_param<std::string>("type");
        GLenum type =
            typeStr == "vertex" ? GL_VERTEX_SHADER :
            typeStr == "fragment" ? GL_FRAGMENT_SHADER :
            GL_FRAGMENT_SHADER;

        auto const &source = get_input<zeno::StringObject>("source")->get();
        auto shader = std::make_shared<GLShaderObject>();
        shader.initialize(type, source);
        set_output("shader", std::move(shader));
    }
};

ZENDEFNODE(GLCreateShader, {
        {"source"},
        {"shader"},
        {{"string", "type", "vertex"}},
        {"EasyGL"},
});

struct GLCreateProgram : zeno::INode {
    virtual void apply() override {
        auto shaderList = get_input<zeno::ListObject>("shaderList");
        auto program = std::make_shared<GLProgramObject>();
        std::vector<GLShaderObject> shaders;
        for (auto const &obj: shaderList->arr) {
            auto shader = zeno::safe_dynamic_cast<GLShaderObject>(obj.get());
            shaders.push_back(*shader);
        }
        program.initialize(shaders);
        set_output("program", std::move(program));
    }
};

ZENDEFNODE(GLCreateProgram, {
        {"shaderList"},
        {"program"},
        {},
        {"EasyGL"},
});

struct GLUseProgram : zeno::INode {
    virtual void apply() override {
        auto program = get_input<GLProgramObject>("program");
        program->use();
    }
};

ZENDEFNODE(GLUseProgram, {
        {"program"},
        {},
        {},
        {"EasyGL"},
});

struct GLDrawArrayTriangles : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<GLPrimitiveObject>("prim");
        std::vector<GLVertexAttribInfo> vabs;
        for (int i = 0; i < prim->attrs(); i++) {
            auto &[arr, dim] = prim->attrs()[i];
            GLVertexAttribInfo vab;
            vab.base = arr.data();
            vab.dim = dim;
            vabs.push_back(vab);
        }
        drawVertexArrays(GL_TRIANGLES, prim->size(), vabs);
        printf("drawing %zd triangle vertices\n", prim->size());
        CHECK_GL(glDrawArrays(GL_TRIANGLES, 0, /*count=*/prim->size()));
    }
};

ZENDEFNODE(GLDrawArrayTriangles, {
        {"prim"},
        {},
        {},
        {"EasyGL"},
});

struct MakeFullscreenRect : zeno::INode {
    inline static const GLfloat vVertices[] = {
        0.0f, 0.0f,
        1.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 1.0f,
    };

    virtual void apply() override {
        auto prim = std::make_shared<GLPrimitiveObject>();
        prim->add_attr(2);
        prim->resize(6);
        std::memcpy(prim->attrs()[0].data(),
                vVertices, sizeof(vVertices));
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(MakeFullscreenRect, {
        {},
        {"prim"},
        {},
        {"EasyGL"},
});

struct GLCreateTextureFramebuffer : zeno::INode {
    virtual void apply() override {
        auto fbo = std::make_shared<GLTextureFramebuffer>();
        auto colorTexList = get_input<zeno::ListObject>("colorTextureList");
        fbo->colorTextures.resize(colorTexList.size());
        for (int i = 0; i < colorTexList.size(); i++) {
            fbo->colorTextures[i].initialize();
        }
        fbo->initialize();
        set_output("framebuffer", std::move(fbo));
    }
};

ZENDEFNODE(GLCreateTextureFramebuffer, {
        {"colorTextureList"},
        {"framebuffer"},
        {},
        {"EasyGL"},
});

struct GLUseFramebuffer : zeno::INode {
    virtual void apply() override {
        auto fbo = get_input<GLFramebufferObject>("framebuffer");
        fbo->use();
    }
};

ZENDEFNODE(GLUseFramebuffer, {
        {"framebuffer"},
        {},
        {},
        {"EasyGL"},
});

struct GLCreateTexture : zeno::INode {
    virtual void apply() override {
        auto tex = std::make_shared<GLTextureObject>();
        tex->initialize();
        set_output("texture", std::move(tex));
    }
};

ZENDEFNODE(GLCreateTexture, {
        {},
        {"texture"},
        {},
        {"EasyGL"},
});

struct GLUseTexture : zeno::INode {
    virtual void apply() override {
        auto tex = get_input<GLTextureObject>("texture");
        auto index = has_input("index") ?
            get_input<zeno::NumericObject>("index")->get<int>() : 0;
        tex->use(index);
    }
};

ZENDEFNODE(GLUseTexture, {
        {"texture", "index"},
        {},
        {},
        {"EasyGL"},
});
