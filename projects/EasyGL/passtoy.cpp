#include <zeno/zeno.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/DictObject.h>
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

static GLfloat generic_triangle_vertices[] = {
    0.0f, 0.0f,
    2.0f, 0.0f,
    0.0f, 2.0f,
};

struct PassToyShader : zeno::IObjectClone<PassToyShader> {
    GLProgramObject prog;
};

struct PassToyMakeShader : zeno::INode {
    inline static std::map<std::string, GLProgramObject> cache;

    virtual void apply() override {
        GLShaderObject frag, vert;
        GLProgramObject prog;

        auto source = get_input<zeno::StringObject>("source")->get();

        if (auto it = cache.find(source); it == cache.end()) {
            zlog::info("compiling shader:\n{}", source);
            frag.initialize(GL_FRAGMENT_SHADER, source);
            vert = get_generic_vertex_shader();
            prog.initialize({vert, frag});
            cache[source] = prog;
        } else {
            prog = it->second;
        }

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


static zeno::vec2i get_default_resolution() {
    zeno::vec2i resolution;
    GLint dims[4];
    glGetIntegerv(GL_VIEWPORT, dims);
    resolution[0] = dims[2] - dims[0];
    resolution[1] = dims[3] - dims[1];
    return resolution;
}

struct PassToyTexture : zeno::IObjectClone<PassToyTexture> {
    GLFramebuffer fbo;
    GLTextureObject tex;
};

static auto make_texture(zeno::vec2i resolution) {
    auto texture = std::make_shared<PassToyTexture>();
    texture->tex.width = resolution[0];
    texture->tex.height = resolution[1];
    texture->tex.type = GL_FLOAT;
    texture->tex.format = GL_RGB;
    texture->tex.internalformat = GL_RGB16F;
    texture->tex.initialize();
    texture->fbo.initialize();
    texture->fbo.bindToTexture(texture->tex, GL_COLOR_ATTACHMENT0);
    texture->fbo.checkStatusComplete();
    return texture;
}

struct PassToyMakeTexture : zeno::INode {
    virtual void apply() override {
        auto resolution = has_input("resolution") ?
            get_input<zeno::NumericObject>("nx")->get<zeno::vec2i>() :
            get_default_resolution();
        auto texture = make_texture(resolution);
        set_output("texture", std::move(texture));
    }
};

ZENDEFNODE(PassToyMakeTexture, {
        {"resolution"},
        {"texture"},
        {},
        {"PassToy"},
});

struct PassToyTexturePair : PassToyTexture {
    PassToyTexture second;

    void swap() {
        std::swap(fbo, second.fbo);
        std::swap(tex, second.tex);
    }
};

struct PassToyMakeTexturePair : zeno::INode {
    virtual void apply() override {
        auto resolution = has_input("resolution") ?
            get_input<zeno::NumericObject>("nx")->get<zeno::vec2i>() :
            get_default_resolution();
        auto texture1 = make_texture(resolution);
        auto texture2 = make_texture(resolution);
        auto texturePair = std::make_shared<PassToyTexturePair>();
        texturePair->fbo = texture1->fbo;
        texturePair->tex = texture1->tex;
        texturePair->second = *texture2;
        set_output("texturePair", std::move(texturePair));
    }
};

ZENDEFNODE(PassToyMakeTexturePair, {
        {"resolution"},
        {"texturePair"},
        {},
        {"PassToy"},
});

struct PassToyTexturePairSwap : zeno::INode {
    virtual void apply() override {
        auto texturePair = get_input<PassToyTexturePair>("texturePair");
        texturePair->swap();
        auto oldTexture = std::make_shared<PassToyTexture>(texturePair->second);
        set_output("texturePair", std::move(texturePair));
        set_output("oldTexture", std::move(oldTexture));
    }
};

ZENDEFNODE(PassToyTexturePairSwap, {
        {"texturePair"},
        {"texturePair", "oldTexture"},
        {},
        {"PassToy"},
});


struct PassToyGetResolution : zeno::INode {
    virtual void apply() override {
        auto resolution = get_default_resolution();
        auto res = std::make_shared<zeno::NumericObject>(resolution);
        set_output("resolution", std::move(res));
    }
};

ZENDEFNODE(PassToyGetResolution, {
        {},
        {"resolution"},
        {},
        {"PassToy"},
});


struct PassToyApplyShader : zeno::INode {
    virtual void apply() override {
        auto resolution = get_default_resolution();

        auto shader = get_input<PassToyShader>("shader");
        shader->prog.use();

        if (has_input("textureIn")) {
            auto textureIns = get_input<zeno::DictObject>("textureIn");
            int i = 0;
            for (auto const &[key, obj]: textureIns->lut) {
                auto textureIn = zeno::safe_dynamic_cast<PassToyTexture>(obj);
                if (i == 0) {
                    resolution = zeno::vec2i(textureIn->tex.width, textureIn->tex.height);
                }
                i += 1;
                //zlog::debug("texture number {} is `{}`", i, key);
                textureIn->tex.use(i);
                shader->prog.setUniform(key.c_str(), i);
            }
        }
        glActiveTexture(GL_TEXTURE0);

        std::shared_ptr<PassToyTexture> textureOut;
        if (has_input<PassToyTexture>("textureOut")) {
            textureOut = get_input<PassToyTexture>("textureOut");
            textureOut->fbo.use();
            resolution[0] = textureOut->tex.width;
            resolution[1] = textureOut->tex.height;
        } else if (!has_input("textureOut")) {
            textureOut = make_texture(resolution);
            textureOut->fbo.use();
        } else {
            GLFramebuffer().use();
        }

        if (has_input("uniforms")) {
            auto uniforms = get_input<zeno::DictObject>("uniforms");
            for (auto const &[key, obj]: uniforms->lut) {
                auto const &value = zeno::safe_dynamic_cast
                    <zeno::NumericObject>(obj)->value;
                std::visit([&shader, key = key] (auto const &value) {
                    shader->prog.setUniform(key.c_str(), value);
                }, value);
            }
        }

        GLVertexAttribInfo vab;
        vab.base = generic_triangle_vertices;
        vab.dim = 2;
        glViewport(0, 0, resolution[0], resolution[1]);
        drawVertexArrays(GL_TRIANGLES, 3, {vab});
        GLFramebuffer().use();
        set_output("textureOut", std::move(textureOut));
    }
};

ZENDEFNODE(PassToyApplyShader, {
        {"shader", "uniforms", "textureIn", "textureOut"},
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
