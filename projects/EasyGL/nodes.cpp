#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/ListObject.h>
#include <GLES2/gl2.h>
#include <cstring>

namespace {

struct GLPrimitiveObject : zeno::IObjectClone<GLPrimitiveObject,
    zeno::PrimitiveObject> {
    std::vector<std::string> boundAttrs;

    void add_bound_attr(std::string const &name) {
        boundAttrs.push_back(name);
    }
};

struct GLShaderObject : zeno::IObjectClone<GLShaderObject> {
    struct Impl {
        GLuint id = 0;
    };

    std::shared_ptr<Impl> impl = std::make_shared<Impl>();

    ~GLShaderObject() {
        if (impl->id)
            glDeleteShader(impl->id);
    }
};

struct GLCreateShader : zeno::INode {
    virtual void apply() override {
        auto typeStr = get_param<std::string>("type");
        GLenum type =
            typeStr == "vertex" ? GL_VERTEX_SHADER :
            typeStr == "fragment" ? GL_FRAGMENT_SHADER :
            GL_FRAGMENT_SHADER;

        auto const &source = get_input<zeno::StringObject>("source")->get();
        const char *sourcePtr = source.c_str();
        auto shader = std::make_shared<GLShaderObject>();
        GLint id = shader->impl->id = glCreateShader(type);
        glShaderSource(id, 1, &sourcePtr, NULL);
        glCompileShader(id);
        GLint status = 0;
        glGetShaderiv(id, GL_COMPILE_STATUS, &status);
        if (!status) {
            GLint infoLen = 0;
            glGetShaderiv(id, GL_INFO_LOG_LENGTH, &infoLen);
            std::vector<char> infoLog(infoLen);
            glGetShaderInfoLog(id, infoLen, NULL, infoLog.data());
            throw zeno::Exception((std::string)infoLog.data());
        }
        set_output("shader", std::move(shader));
    }
};

ZENDEFNODE(GLCreateShader, {
        {"source"},
        {"shader"},
        {{"string", "type", "vertex"}},
        {"EasyGL"},
});

struct GLProgramObject : zeno::IObjectClone<GLProgramObject> {
    struct Impl {
        GLuint id = 0;
    };

    std::shared_ptr<Impl> impl = std::make_shared<Impl>();

    ~GLProgramObject() {
        if (impl->id)
            glDeleteProgram(impl->id);
    }

    operator GLuint() const {
        return impl->id;
    }
};

struct GLCreateProgram : zeno::INode {
    virtual void apply() override {
        auto shaderList = get_input<zeno::ListObject>("shaderList");
        auto program = std::make_shared<GLProgramObject>();
        GLint id = program->impl->id = glCreateProgram();
        for (auto const &obj: shaderList->arr) {
            auto shader = zeno::safe_dynamic_cast<GLShaderObject>(obj.get());
            glAttachShader(id, shader->impl->id);
        }
        glLinkProgram(id);
        GLint status = 0;
        glGetProgramiv(id, GL_LINK_STATUS, &status);
        if (!status) {
            GLint infoLen = 0;
            glGetProgramiv(id, GL_INFO_LOG_LENGTH, &infoLen);
            if (infoLen) {
                std::vector<char> infoLog(infoLen);
                glGetProgramInfoLog(id, infoLen, NULL, infoLog.data());
                throw zeno::Exception((std::string)infoLog.data());
            }
        }
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
        glUseProgram(program->impl->id);
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
        for (int i = 0; i < prim->boundAttrs.size(); i++) {
            auto name = prim->boundAttrs[i];
            std::visit([&] (auto const &arr) {
                using T = std::decay_t<decltype(arr[0])>;
                using S = zeno::decay_vec_t<T>;
                glEnableVertexAttribArray(i);
                glVertexAttribPointer
                        ( /*index=*/i
                        , /*size=*/zeno::is_vec_n<T>
                        , GL_FLOAT
                        , GL_FALSE
                        , /*stride=*/0
                        , (void *)arr.data()
                        );
            }, prim->attr(name));
        }
        printf("drawing %zd triangle vertices\n", prim->size());
        glDrawArrays(GL_TRIANGLES, 0, /*count=*/prim->size());
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
        -1.0f, -1.0f, 0.0f,
         1.0f, -1.0f, 0.0f,
        -1.0f,  1.0f, 0.0f,
         1.0f, -1.0f, 0.0f,
        -1.0f, -1.0f, 0.0f,
         1.0f,  1.0f, 0.0f,
    };

    virtual void apply() override {
        auto prim = std::make_shared<GLPrimitiveObject>();
        prim->add_attr<zeno::vec3f>("pos");
        prim->add_bound_attr("pos");
        prim->resize(6);
        std::memcpy(prim->attr<zeno::vec3f>("pos").data(),
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

struct GLFramebufferObject : zeno::IObjectClone<GLFramebufferObject> {
    struct Impl {
        GLuint id = 0;
    };

    std::shared_ptr<Impl> impl = std::make_shared<Impl>();

    ~GLFramebufferObject() {
        if (impl->id)
            glDeleteFramebuffers(1, &impl->id);
    }

    operator GLuint() const {
        return impl->id;
    }
};

struct GLCreateFramebuffer : zeno::INode {
    virtual void apply() override {
        auto fbo = std::make_shared<GLFramebufferObject>();
        glGenFramebuffers(1, &fbo->impl->id);
        set_output("framebuffer", std::move(fbo));
    }
};

ZENDEFNODE(GLCreateFramebuffer, {
        {},
        {"framebuffer"},
        {},
        {"EasyGL"},
});

struct GLTextureObject : zeno::IObjectClone<GLTextureObject> {
    struct Impl {
        GLuint id = 0;
    };

    std::shared_ptr<Impl> impl = std::make_shared<Impl>();

    ~GLTextureObject() {
        if (impl->id)
            glDeleteTextures(1, &impl->id);
    }

    operator GLuint() const {
        return impl->id;
    }
};

struct GLCreateTexture : zeno::INode {
    virtual void apply() override {
        auto tex = std::make_shared<GLTextureObject>();
        glGenTextures(1, &tex->impl->id);
        set_output("texture", std::move(tex));
    }
};

ZENDEFNODE(GLCreateTexture, {
        {},
        {"texture"},
        {},
        {"EasyGL"},
});

struct GLBindFramebufferTexture : zeno::INode {
    virtual void apply() override {
        auto fbo = get_input<GLFramebufferObject>("framebuffer");
        auto tex = get_input<GLTextureObject>("texture");
        glBindFramebuffer(GL_FRAMEBUFFER, fbo->impl->id);
        glBindTexture(GL_TEXTURE_2D, tex->impl->id);
        glFramebufferTexture2D(GL_FRAMEBUFFER,
                GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex->impl->id, 0);
    }
};

ZENDEFNODE(GLBindFramebufferTexture, {
        {"framebuffer", "texture"},
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
    inline static const GLfloat vVelocities[] = {
        -0.5f, -0.3f, 0.0f,
        0.2f, -0.6f, 0.0f,
        0.1f,  0.7f, 0.0f,
    };

    virtual void apply() override {
        auto prim = std::make_shared<GLPrimitiveObject>();
        prim->add_attr<zeno::vec3f>("pos");
        prim->add_attr<zeno::vec3f>("vel");
        prim->add_bound_attr("pos");
        prim->resize(3);
        std::memcpy(prim->attr<zeno::vec3f>("pos").data(),
                vVertices, sizeof(vVertices));
        std::memcpy(prim->attr<zeno::vec3f>("vel").data(),
                vVelocities, sizeof(vVelocities));
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(MakeSimpleTriangle, {
        {},
        {"prim"},
        {},
        {"EasyGL"},
});

struct DemoAdvectParticlesInBox : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<GLPrimitiveObject>("prim");
        float dt = get_input<zeno::NumericObject>("dt")->get<float>();
        auto &pos = prim->attr<zeno::vec3f>("pos");
        auto &vel = prim->attr<zeno::vec3f>("vel");
        for (int i = 0; i < prim->size(); i++) {
            pos[i] += vel[i] * dt;
            for (int j = 0; j < 3; j++) {
                if (pos[i][j] > 1 && vel[i][j] > 0) {
                    vel[i][j] = -vel[i][j];
                } else if (pos[i][j] < -1 && vel[i][j] < 0) {
                    vel[i][j] = -vel[i][j];
                }
            }
        }
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(DemoAdvectParticlesInBox, {
        {"prim", "dt"},
        {"prim"},
        {},
        {"EasyGL"},
});

}
