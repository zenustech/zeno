#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/ListObject.h>
#include "GLVertexAttribInfo.h"
#include "GLPrimitiveObject.h"
#include "GLTextureObject.h"
#include "GLShaderObject.h"
#include "GLProgramObject.h"
#include "GLFramebuffer.h"

namespace {

struct GLCreateShader : zeno::INode {
    virtual void apply() override {
        auto typeStr = get_param<std::string>("type");
        GLenum type =
            typeStr == "vertex" ? GL_VERTEX_SHADER :
            typeStr == "fragment" ? GL_FRAGMENT_SHADER :
            GL_FRAGMENT_SHADER;

        auto const &source = get_input<zeno::StringObject>("source")->get();
        auto shader = std::make_shared<GLShaderObject>();
        shader->initialize(type, source);
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
        for (auto &&obj: shaderList->get()) {
            auto shader = zeno::safe_dynamic_cast<GLShaderObject>(obj.get());
            shaders.push_back(*shader);
        }
        program->initialize(shaders);
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
        for (int i = 0; i < prim->nattrs(); i++) {
            auto &[arr, dim] = prim->attr(i);
            GLVertexAttribInfo vab;
            vab.base = (void *)arr.data();
            vab.dim = dim;
            vabs.push_back(vab);
        }
        printf("drawing %zd triangle vertices\n", prim->size());
        drawVertexArrays(GL_TRIANGLES, prim->size(), vabs);
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
        0.0f, 0.5f,
        -0.5f, -0.5f,
        0.5f, -0.5f,
    };

    virtual void apply() override {
        auto prim = std::make_shared<GLPrimitiveObject>();
        prim->add_attr(2);
        prim->resize(3);
        std::memcpy(prim->attr(0).arr.data(), vVertices, sizeof(vVertices));
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(MakeSimpleTriangle, {
        {},
        {"prim"},
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
        std::memcpy(prim->attr(0).arr.data(), vVertices, sizeof(vVertices));
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(MakeFullscreenRect, {
        {},
        {"prim"},
        {},
        {"EasyGL"},
});

struct GLNoFramebuffer : zeno::INode {
    virtual void apply() override {
        auto fbo = std::make_shared<GLFramebuffer>();
        set_output("framebuffer", std::move(fbo));
    }
};

ZENDEFNODE(GLNoFramebuffer, {
        {},
        {"framebuffer"},
        {},
        {"EasyGL"},
});

struct GLCreateFramebuffer : zeno::INode {
    virtual void apply() override {
        auto fbo = std::make_shared<GLFramebuffer>();
        fbo->initialize();
        set_output("framebuffer", std::move(fbo));
    }
};

ZENDEFNODE(GLCreateFramebuffer, {
        {},
        {"framebuffer"},
        {},
        {"EasyGL"},
});

struct GLTexturedFramebuffer : GLFramebuffer {
    GLTextureObject tex;
};

struct GLBindFramebufferTexture : zeno::INode {
    virtual void apply() override {
        auto fbo = get_input<GLFramebuffer>("framebuffer");
        auto tex = get_input<GLTextureObject>("texture");
        fbo->bindToTexture(*tex, GL_COLOR_ATTACHMENT0);
        fbo->checkStatusComplete();
        auto new_fbo = std::make_shared<GLTexturedFramebuffer>();
        new_fbo->impl = fbo->impl;
        new_fbo->tex = *tex;
        set_output("framebuffer", std::move(new_fbo));
    }
};

ZENDEFNODE(GLBindFramebufferTexture, {
        {"framebuffer", "texture"},
        {"framebuffer"},
        {},
        {"EasyGL"},
});

struct GLGetFramebufferTexture : zeno::INode {
    virtual void apply() override {
        auto fbo = get_input<GLTexturedFramebuffer>("framebuffer");
        auto tex = std::make_shared<GLTextureObject>(fbo->tex);
        set_output("texture", std::move(tex));
    }
};

ZENDEFNODE(GLGetFramebufferTexture, {
        {"framebuffer"},
        {"texture"},
        {},
        {"EasyGL"},
});

struct GLUseFramebuffer : zeno::INode {
    virtual void apply() override {
        if (has_input("framebuffer")) {
            auto fbo = get_input<GLFramebuffer>("framebuffer");
            fbo->use();
        } else {
            GLFramebuffer().use();
        }
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

struct GLClearColor : zeno::INode {
    virtual void apply() override {
        auto color = has_input("color") ?
            get_input<zeno::NumericObject>("color")->get<zeno::vec3f>() :
            zeno::vec3f(0);
        auto alpha = has_input("alpha") ?
            get_input<zeno::NumericObject>("alpha")->get<float>() :
            0.0f;
        glClearColor(color[0], color[1], color[2], alpha);
        glClear(GL_COLOR_BUFFER_BIT);
    }
};

ZENDEFNODE(GLClearColor, {
        {"color", "alpha"},
        {},
        {},
        {"EasyGL"},
});

}
