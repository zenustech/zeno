#include <zeno/zeno.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/DictObject.h>
#include <zeno/types/VoidPtrObject.h>
#include "GLVertexAttribInfo.h"
#include "GLPrimitiveObject.h"
#include "GLTextureObject.h"
#include "GLShaderObject.h"
#include "GLProgramObject.h"
#include "GLFramebuffer.h"
#include "stb_image.h"
#include <algorithm>

namespace {


static GLShaderObject get_generic_vertex_shader() {
    static GLShaderObject vert;
    if (!vert.impl) {
        vert.initialize(GL_VERTEX_SHADER, R"GLSL(#version 300 es
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
            //zlog::info("compiling shader:\n{}", source);
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

struct PassToyDownscaleResolution : zeno::INode {
    virtual void apply() override {
        auto scale = get_input<zeno::NumericObject>("scale")->get<int>();
        auto res = get_default_resolution();
        res /= scale;
        glViewport(0, 0, res[0], res[1]);
    }
};

ZENDEFNODE(PassToyDownscaleResolution, {
        {"scale"},
        {},
        {},
        {"PassToy"},
});

struct PassToyUpscaleResolution : zeno::INode {
    virtual void apply() override {
        auto scale = get_input<zeno::NumericObject>("scale")->get<int>();
        auto res = get_default_resolution();
        res *= scale;
        glViewport(0, 0, res[0], res[1]);
    }
};

ZENDEFNODE(PassToyUpscaleResolution, {
        {"scale"},
        {},
        {},
        {"PassToy"},
});


struct PassToyTexture : zeno::IObjectClone<PassToyTexture> {
    GLFramebuffer fbo;
    GLTextureObject tex;
};

static std::tuple<GLenum, GLenum, GLenum>
internalformat_from_string(std::string name) {
    std::transform(name.begin(), name.end(), name.begin(), ::toupper);
    if (0) {
#define _EVAL(x) x
#define _PER_FMT(x, y, z) \
    } else if (name == #x) { \
        return {_EVAL(GL_##x), _EVAL(GL_##y), _EVAL(GL_##z)};

    _PER_FMT(RED, RED, UNSIGNED_BYTE)
    _PER_FMT(R8UI, RED, UNSIGNED_BYTE)
    _PER_FMT(R16UI, RED, UNSIGNED_SHORT)
    _PER_FMT(R32UI, RED, UNSIGNED_INT)
    _PER_FMT(R8I, RED, BYTE)
    _PER_FMT(R16I, RED, SHORT)
    _PER_FMT(R32I, RED, INT)
    _PER_FMT(R16F, RED, HALF_FLOAT)
    _PER_FMT(R32F, RED, FLOAT)

    _PER_FMT(RG, RG, UNSIGNED_BYTE)
    _PER_FMT(RG8UI, RG, UNSIGNED_BYTE)
    _PER_FMT(RG16UI, RG, UNSIGNED_SHORT)
    _PER_FMT(RG32UI, RG, UNSIGNED_INT)
    _PER_FMT(RG8I, RG, BYTE)
    _PER_FMT(RG16I, RG, SHORT)
    _PER_FMT(RG32I, RG, INT)
    _PER_FMT(RG16F, RG, HALF_FLOAT)
    _PER_FMT(RG32F, RG, FLOAT)

    _PER_FMT(RGB, RGB, UNSIGNED_BYTE)
    _PER_FMT(RGB8UI, RGB, UNSIGNED_BYTE)
    _PER_FMT(RGB16UI, RGB, UNSIGNED_SHORT)
    _PER_FMT(RGB32UI, RGB, UNSIGNED_INT)
    _PER_FMT(RGB8I, RGB, BYTE)
    _PER_FMT(RGB16I, RGB, SHORT)
    _PER_FMT(RGB32I, RGB, INT)
    _PER_FMT(RGB16F, RGB, HALF_FLOAT)
    _PER_FMT(RGB32F, RGB, FLOAT)

    _PER_FMT(RGBA, RGBA, UNSIGNED_BYTE)
    _PER_FMT(RGBA8UI, RGBA, UNSIGNED_BYTE)
    _PER_FMT(RGBA16UI, RGBA, UNSIGNED_SHORT)
    _PER_FMT(RGBA32UI, RGBA, UNSIGNED_INT)
    _PER_FMT(RGBA8I, RGBA, BYTE)
    _PER_FMT(RGBA16I, RGBA, SHORT)
    _PER_FMT(RGBA32I, RGBA, INT)
    _PER_FMT(RGBA16F, RGBA, HALF_FLOAT)
    _PER_FMT(RGBA32F, RGBA, FLOAT)

#undef _PER_FMT
#undef _EVAL
    }
//    zlog::error("bad format string {}", name);
    abort();
}

static auto make_texture(zeno::vec2i resolution, std::string const &format) {
    auto texture = std::make_shared<PassToyTexture>();
    texture->tex.width = resolution[0];
    texture->tex.height = resolution[1];
    auto [ifmt, fmt, typ] = internalformat_from_string(format);
    texture->tex.internalformat = ifmt;
    texture->tex.format = fmt;
    texture->tex.type = typ;
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
        auto format = get_param<std::string>("format");
        auto texture = make_texture(resolution, format);
        set_output("texture", std::move(texture));
    }
};

ZENDEFNODE(PassToyMakeTexture, {
        {"resolution"},
        {"texture"},
        {{"string", "format", "rgb16f"}},
        {"PassToy"},
});

struct PassToyGetTextureInteger : zeno::INode {
    virtual void apply() override {
        auto texture = get_input<PassToyTexture>("texture");
        auto id = std::make_shared<zeno::NumericObject>();
        id->value = (int)texture->tex.impl->id;
        set_output("id", std::move(id));
    }
};

ZENDEFNODE(PassToyGetTextureInteger, {
        {"texture"},
        {"id"},
        {},
        {"PassToy"},
});

struct PassToyImageTextureFromVoidPtr : zeno::INode {
    virtual void apply() override {
        void *p = get_input<zeno::VoidPtrObject>("voidPtr")->get();
        auto texture = std::make_shared<PassToyTexture>();
        // zlog::info("loading image file from void ptr {}", p);
        auto nx = 0[(int *)p];
        auto ny = 1[(int *)p];
        auto img = (unsigned char *)p + 8;
        for (int i = 0; i < nx * ny * 4; i += 4) {
            std::swap(img[i + 0], img[i + 2]);
        }
        // zlog::info("loaded {}x{} at {}", nx, ny, (void *)img);
        texture->tex.width = nx;
        texture->tex.height = ny;
        texture->tex.type = GL_UNSIGNED_BYTE;
        texture->tex.format = GL_RGBA;
        texture->tex.internalformat = GL_RGBA;
        texture->tex.base = img;
        texture->tex.initialize();
        texture->fbo.initialize();
        texture->fbo.bindToTexture(texture->tex, GL_COLOR_ATTACHMENT0);
        texture->fbo.checkStatusComplete();
        set_output("texture", std::move(texture));
    }
};

ZENDEFNODE(PassToyImageTextureFromVoidPtr, {
        {"voidPtr"},
        {"texture"},
        {},
        {"PassToy"},
});

struct PassToyImageTextureFromVoidPtrAndRes : zeno::INode {
    virtual void apply() override {
        void *p = get_input<zeno::VoidPtrObject>("voidPtr")->get();
        auto res = get_input<zeno::NumericObject>("voidPtr")->get<zeno::vec2i>();
        auto nx = res[0], ny = res[1];
        auto texture = std::make_shared<PassToyTexture>();
        // zlog::info("loading image file from void ptr {}", p);
        // >>> tianjia zhexie daima
        auto img = (unsigned char *)p + 8;
        for (int i = 0; i < nx * ny * 4; i += 4) {
            std::swap(img[i + 0], img[i + 2]);
        }
        // <<< tianjia zhexie daima
        // zlog::info("loaded {}x{} at {}", nx, ny, p);
        texture->tex.width = nx;
        texture->tex.height = ny;
        texture->tex.type = GL_UNSIGNED_BYTE;
        texture->tex.format = GL_RGBA;
        texture->tex.internalformat = GL_RGBA;
        texture->tex.base = p;
        texture->tex.initialize();
        texture->fbo.initialize();
        texture->fbo.bindToTexture(texture->tex, GL_COLOR_ATTACHMENT0);
        texture->fbo.checkStatusComplete();
        set_output("texture", std::move(texture));
    }
};

ZENDEFNODE(PassToyImageTextureFromVoidPtrAndRes, {
        {"voidPtr", "resolution"},
        {"texture"},
        {},
        {"PassToy"},
});


struct PassToyLoadImageTexture : zeno::INode {
    virtual void apply() override {
        auto path = get_param<std::string>("path");
        auto texture = std::make_shared<PassToyTexture>();
        int nx = 0, ny = 0, nc = 0;
        //stbi_set_flip_vertically_on_load(true);
        // zlog::info("loading image file: {}", path);
        unsigned char *img = stbi_load(path.c_str(), &nx, &ny, &nc, 0);
        // zlog::info("loaded {}x{}x{} at {}", nx, ny, nc, (void *)img);
        int format = GL_RGB;
        switch (nc) {
        case 4: format = GL_RGBA; break;
        case 3: format = GL_RGB; break;
        case 2: format = GL_RG; break;
        case 1: format = GL_RED; break;
        };
        texture->tex.width = nx;
        texture->tex.height = ny;
        texture->tex.type = GL_UNSIGNED_BYTE;
        texture->tex.format = format;
        texture->tex.internalformat = format;
        //texture->tex.minFilter = GL_LINEAR_MIPMAP_LINEAR;
        //texture->tex.magFilter = GL_LINEAR_MIPMAP_LINEAR;
        texture->tex.base = img;
        texture->tex.initialize();
        texture->fbo.initialize();
        texture->fbo.bindToTexture(texture->tex, GL_COLOR_ATTACHMENT0);
        texture->fbo.checkStatusComplete();
        if (img)
            stbi_image_free(img);
        set_output("texture", std::move(texture));
    }
};

ZENDEFNODE(PassToyLoadImageTexture, {
        {},
        {"texture"},
        {{"string", "path", ""}},
        {"PassToy"},
});

struct PassToyGetTextureResolution : zeno::INode {
    virtual void apply() override {
        auto texture = get_input<PassToyTexture>("texture");
        zeno::vec2i resolution(texture->tex.width, texture->tex.height);
        auto res = std::make_shared<zeno::NumericObject>(resolution);
        set_output("resolution", std::move(res));
    }
};

ZENDEFNODE(PassToyGetTextureResolution, {
        {"texture"},
        {"resolution"},
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
        auto format = get_param<std::string>("format");
        auto texture1 = make_texture(resolution, format);
        auto texture2 = make_texture(resolution, format);
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
        {{"string", "format", "rgb16f"}},
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



struct PassToyApplyShader : zeno::INode {
    virtual void apply() override {
        auto resolution = get_default_resolution();

        auto shader = get_input<PassToyShader>("shader");
        shader->prog.use();

        if (has_input("textureIn")) {
            auto textureIns = get_input<zeno::DictObject>("textureIn");
            int i = 0;
            for (auto const &[key, obj]: textureIns->lut) {
                auto textureIn = zeno::smart_any_cast<std::shared_ptr<PassToyTexture>>(obj);
                i += 1;
                //zlog::debug("texture number {} is `{}`", i, key);
                textureIn->tex.use(i);
                shader->prog.setUniform(key.c_str(), i);
            }
            //zlog::debug("done assigning input textures");
        }
        glActiveTexture(GL_TEXTURE0);

        std::shared_ptr<PassToyTexture> textureOut;
        if (has_input<PassToyTexture>("textureOut")) {
            textureOut = get_input<PassToyTexture>("textureOut");
            textureOut->fbo.use();
        } else if (!has_input("textureOut")) {
            textureOut = make_texture(resolution, "rgb16f");
            textureOut->fbo.use();
        } else {
            GLFramebuffer().use();
        }

        if (has_input("uniforms")) {
            auto uniforms = get_input<zeno::DictObject>("uniforms");
            for (auto const &[key, obj]: uniforms->lut) {
                auto const &value = zeno::smart_any_cast<std::shared_ptr<zeno::NumericObject>>(obj)->value;
                std::visit([&shader, key = key] (auto const &value) {
                    shader->prog.setUniform(key.c_str(), value);
                }, value);
            }
        }

        GLVertexAttribInfo vab;
        vab.base = generic_triangle_vertices;
        vab.dim = 2;
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
