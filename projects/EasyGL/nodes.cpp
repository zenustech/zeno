#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
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
        GLint id = 0;
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
        GLint id = 0;
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
            auto shader = dynamic_cast<GLShaderObject *>(obj.get());
            if (shader)
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
                printf("%zd\n", zeno::is_vec_n<T>);
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
        prim->resize(3);
        std::memcpy(prim->attr<zeno::vec3f>("pos").data(),
                vVertices, sizeof(vVertices));
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
