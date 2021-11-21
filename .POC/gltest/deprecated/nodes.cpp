#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/ListObject.h>
#include <GLES2/gl2.h>
#include <cstring>

namespace {

  static const char *get_opengl_error_string(GLenum err) {
    switch (err) {
#define PER_GL_ERR(x) \
  case x:             \
    return #x;
      PER_GL_ERR(GL_NO_ERROR)
      PER_GL_ERR(GL_INVALID_ENUM)
      PER_GL_ERR(GL_INVALID_VALUE)
      PER_GL_ERR(GL_INVALID_OPERATION)
      PER_GL_ERR(GL_INVALID_FRAMEBUFFER_OPERATION)
      PER_GL_ERR(GL_OUT_OF_MEMORY)
#undef PER_GL_ERR
    }
    static char tmp[233];
    sprintf(tmp, "%d\n", err);
    return tmp;
  }

  static void _check_opengl_error(const char *file, int line, const char *hint) {
    auto err = glGetError();
    if (err != GL_NO_ERROR) {
      auto msg = get_opengl_error_string(err);
      printf("%s:%d: `%s`: %s\n", file, line, hint, msg);
      abort();
    }
  }

#define CHECK_GL(x) do { (x); \
    _check_opengl_error(__FILE__, __LINE__, #x); \
  } while (0)

struct GLPrimitiveObject : zeno::IObjectClone<GLPrimitiveObject> {
    std::vector<std::vector<float>> attrs;
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

        ~Impl() {
            if (id)
                CHECK_GL(glDeleteProgram(id));
        }
    };

    std::shared_ptr<Impl> impl = std::make_shared<Impl>();

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
        CHECK_GL(glUseProgram(program->impl->id));
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
                CHECK_GL(glEnableVertexAttribArray(i));
                CHECK_GL(glVertexAttribPointer
                        ( /*index=*/i
                        , /*size=*/zeno::is_vec_n<T>
                        , GL_FLOAT
                        , GL_FALSE
                        , /*stride=*/0
                        , (void *)arr.data()
                        ));
            }, prim->attr(name));
        }
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
        0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        1.0f, 1.0f, 0.0f,
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

        ~Impl() {
            if (id)
                CHECK_GL(glDeleteFramebuffers(1, &id));
        }
    };

    std::shared_ptr<Impl> impl = std::make_shared<Impl>();

    operator GLuint() const {
        return impl->id;
    }
};

struct GLCreateFramebuffer : zeno::INode {
    virtual void apply() override {
        auto fbo = std::make_shared<GLFramebufferObject>();
        CHECK_GL(glGenFramebuffers(1, &fbo->impl->id));
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

        ~Impl() {
            if (id)
                CHECK_GL(glDeleteTextures(1, &id));
        }
    };

    std::shared_ptr<Impl> impl = std::make_shared<Impl>();

    operator GLuint() const {
        return impl->id;
    }
};

struct GLCreateTexture : zeno::INode {
    virtual void apply() override {
        auto tex = std::make_shared<GLTextureObject>();
        CHECK_GL(glGenTextures(1, &tex->impl->id));
        CHECK_GL(glBindTexture(GL_TEXTURE_2D, tex->impl->id));
        CHECK_GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
        CHECK_GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
        CHECK_GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
        CHECK_GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
        CHECK_GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 512, 512, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL));
        set_output("texture", std::move(tex));
    }
};

ZENDEFNODE(GLCreateTexture, {
        {},
        {"texture"},
        {},
        {"EasyGL"},
});

struct GLBindTexture : zeno::INode {
    virtual void apply() override {
        auto tex = get_input<GLTextureObject>("texture");
        int num = has_input("number") ?
            get_input<zeno::NumericObject>("number")->get<int>() :
            0;
        CHECK_GL(glActiveTexture(GL_TEXTURE0 + num));
        CHECK_GL(glBindTexture(GL_TEXTURE_2D, tex->impl->id));
    }
};

ZENDEFNODE(GLBindTexture, {
        {"texture", "number"},
        {},
        {},
        {"EasyGL"},
});

struct GLBindFramebufferTexture : zeno::INode {
    virtual void apply() override {
        auto fbo = get_input<GLFramebufferObject>("framebuffer");
        auto tex = get_input<GLTextureObject>("texture");
        CHECK_GL(glBindFramebuffer(GL_FRAMEBUFFER, fbo->impl->id));
        CHECK_GL(glBindTexture(GL_TEXTURE_2D, tex->impl->id));
        CHECK_GL(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex->impl->id, 0));
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            printf("ERROR: framebuffer status is not complete!\n");
        }
    }
};

ZENDEFNODE(GLBindFramebufferTexture, {
        {"framebuffer", "texture"},
        {},
        {},
        {"EasyGL"},
});

struct GLUnbindFramebuffer : zeno::INode {
    virtual void apply() override {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
};

ZENDEFNODE(GLUnbindFramebuffer, {
        {},
        {},
        {},
        {"EasyGL"},
});

struct GLViewportSize : zeno::INode {
    virtual void apply() override {
        auto bx = has_input("baseX") ?
            get_input<zeno::NumericObject>("baseX")->get<float>() :
            0;
        auto by = has_input("baseY") ?
            get_input<zeno::NumericObject>("baseY")->get<float>() :
            0;
        auto nx = has_input("sizeX") ?
            get_input<zeno::NumericObject>("sizeX")->get<float>() :
            0;
        auto ny = has_input("sizeY") ?
            get_input<zeno::NumericObject>("sizeY")->get<float>() :
            0;
    }
};

ZENDEFNODE(GLViewportSize, {
        {"baseX", "baseY", "sizeX", "sizeY"},
        {},
        {},
        {"EasyGL"},
});

struct GLClearColor : zeno::INode {
    virtual void apply() override {
        auto rgb = has_input("color") ?
            get_input<zeno::NumericObject>("color")->get<zeno::vec3f>() :
            zeno::vec3f(0);
        auto a = has_input("alpha") ?
            get_input<zeno::NumericObject>("alph")->get<float>() :
            0.0f;
        glClearColor(rgb[0], rgb[1], rgb[2], a);
        glClear(GL_COLOR_BUFFER_BIT);
    }
};

ZENDEFNODE(GLClearColor, {
        {"color", "alpha"},
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
