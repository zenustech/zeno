#include <zeno/zeno.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/ListObject.h>
#include <GLES2/gl2.h>
#include <cstring>
#include "GLVertexAttribInfo.h"
#include "GLPrimitiveObject.h"
#include "GLTextureObject.h"
#include "GLShaderObject.h"
#include "GLProgramObject.h"
#include "GLFramebuffer.h"

namespace {


static GLShaderObject get_generic_vertex_shader() {
    static GLShaderObject vert;
    if (!vert.impl) {
        vert.initialize(GL_VERTEX_SHADER, R"GLSL(
#version 300 es
layout (location = 0) in vec2 vPosition;
out vec2 fragCoord;
void main() {
  fragCoord = vPosition;
  gl_Position = vec4(vPosition * 2.0 - 1.0, 0.0, 1.0);
}
)GLSL");
    }
    return vert;
}

static GLfloat generic_rect_vertices[] = {
    0.0f, 0.0f,
    1.0f, 0.0f,
    0.0f, 1.0f,
    1.0f, 0.0f,
    0.0f, 1.0f,
    1.0f, 1.0f,
};

struct PassToyShader : zeno::IObjectClone<PassToyShader> {
    GLProgramObject prog;
};

struct PassToyMakeShader : zeno::INode {
    virtual void apply() override {
        GLShaderObject frag, vert;
        GLProgramObject prog;

        auto source = get_input<zeno::StringObject>("source")->get();
        frag.initialize(GL_FRAGMENT_SHADER, source);
        vert = get_generic_vertex_shader();
        prog.initialize({vert, frag});

        auto shader = std::make_shared<PassToyShader>();
        shader->prog = prog;
        set_output("shader", std::move(shader));
    }
};

ZENDEFNODE(PassToyMakeShader, {
        {"source"},
        {"shader"},
        {},
        {"PassToy"},
});


struct PassToyTexture : zeno::IObjectClone<PassToyTexture> {
    GLFramebuffer fbo;
    GLTextureObject tex;
};

struct PassToyMakeTexture : zeno::INode {
    virtual void apply() override {
        auto nx = get_input<zeno::NumericObject>("nx")->get<int>();
        auto ny = get_input<zeno::NumericObject>("ny")->get<int>();
        zeno::vec2i resolution(nx, ny);
        auto texture = std::make_shared<PassToyTexture>();
        texture->tex.width = resolution[0];
        texture->tex.height = resolution[1];
        texture->tex.initialize();
        texture->fbo.initialize();
        texture->fbo.bindToTexture(texture->tex, GL_COLOR_ATTACHMENT0);
        texture->fbo.checkStatusComplete();
        set_output("texture", std::move(texture));
    }
};

ZENDEFNODE(PassToyMakeTexture, {
        {"nx", "ny"},
        {"texture"},
        {},
        {"PassToy"},
});


struct PassToyApplyShader : zeno::INode {
    virtual void apply() override {
        zeno::vec2i resolution(512, 512);
        if (has_input<zeno::ListObject>("textureIn")) {
            auto textureInList = get_input<zeno::ListObject>("textureIn");
            for (int i = 0; i < textureInList->arr.size(); i++) {
                auto obj = textureInList->arr[i].get();
                auto textureIn = zeno::safe_dynamic_cast<PassToyTexture>(obj);
                textureIn->tex.use(i);
                if (i == 0)
                    resolution = zeno::vec2i(
                            textureIn->tex.width, textureIn->tex.height);
            }
        } else if (has_input("textureIn")) {
            auto textureIn = get_input<PassToyTexture>("textureIn");
            textureIn->tex.use(0);
            resolution = zeno::vec2i(
                    textureIn->tex.width, textureIn->tex.height);
        }

        std::shared_ptr<PassToyTexture> textureOut;
        if (has_input<PassToyTexture>("textureOut")) {
            textureOut = get_input<PassToyTexture>("textureOut");
            textureOut->fbo.use();
        } else if (!has_input("textureOut")) {
            textureOut = std::make_shared<PassToyTexture>();
            textureOut->tex.width = resolution[0];
            textureOut->tex.height = resolution[1];
            textureOut->tex.initialize();
            textureOut->fbo.initialize();
            textureOut->fbo.bindToTexture(textureOut->tex, GL_COLOR_ATTACHMENT0);
            textureOut->fbo.checkStatusComplete();
            textureOut->fbo.use();
        } else {
            GLFramebuffer().use();
        }

        auto shader = get_input<PassToyShader>("shader");
        shader->prog.use();
        GLVertexAttribInfo vab;
        vab.base = generic_rect_vertices;
        vab.dim = 2;
        drawVertexArrays(GL_TRIANGLES, 6, {vab});
        GLFramebuffer().use();
        set_output("textureOut", std::move(textureOut));
    }
};

ZENDEFNODE(PassToyApplyShader, {
        {"shader", "textureIn", "textureOut"},
        {"textureOut"},
        {},
        {"PassToy"},
});

struct PassToyScreenTexture : zeno::INode {
    virtual void apply() override {
        set_output("texture", std::make_shared<zeno::StringObject>());
    }
};

ZENDEFNODE(PassToyScreenTexture, {
        {},
        {"texture"},
        {},
        {"PassToy"},
});


}
