#pragma once

#include "common.h"
#include "GLShaderObject.h"
#include <zeno/core/IObject.h>
#include <zeno/utils/vec.h>

struct GLProgramObject : zeno::IObjectClone<GLProgramObject> {
    struct Impl {
        GLuint id = 0;
        ~Impl() { if (id) glDeleteProgram(id); }
    };
    std::shared_ptr<Impl> impl;
    operator GLuint() { return impl->id; }

    void initialize(std::vector<GLShaderObject> const &shaders) {
        GLint id = glCreateProgram();
        for (auto const &shader: shaders) {
            glAttachShader(id, shader.impl->id);
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
                //zlog::error("error linking program: {}", infoLog.data());
            }
        }
        impl = std::make_shared<Impl>();
        impl->id = id;
    }

    void use() {
        CHECK_GL(glUseProgram(impl->id));
    }

    void setUniform(int loc, int x) {
        CHECK_GL(glUniform1i(loc, x));
    }

    void setUniform(int loc, zeno::vec2i const &x) {
        CHECK_GL(glUniform2i(loc, x[0], x[1]));
    }

    void setUniform(int loc, zeno::vec3i const &x) {
        CHECK_GL(glUniform3i(loc, x[0], x[1], x[2]));
    }

    void setUniform(int loc, zeno::vec4i const &x) {
        CHECK_GL(glUniform4i(loc, x[0], x[1], x[2], x[3]));
    }

    void setUniform(int loc, float x) {
        CHECK_GL(glUniform1f(loc, x));
    }

    void setUniform(int loc, zeno::vec2f const &x) {
        CHECK_GL(glUniform2f(loc, x[0], x[1]));
    }

    void setUniform(int loc, zeno::vec3f const &x) {
        CHECK_GL(glUniform3f(loc, x[0], x[1], x[2]));
    }

    void setUniform(int loc, zeno::vec4f const &x) {
        CHECK_GL(glUniform4f(loc, x[0], x[1], x[2], x[3]));
    }

    void setUniform(int loc, bool x) {
        CHECK_GL(glUniform1i(loc, x));
    }

    void setUniform(int loc, zeno::vec2b const &x) {
        CHECK_GL(glUniform2i(loc, x[0], x[1]));
    }

    void setUniform(int loc, zeno::vec3b const &x) {
        CHECK_GL(glUniform3i(loc, x[0], x[1], x[2]));
    }

    void setUniform(int loc, zeno::vec4b const &x) {
        CHECK_GL(glUniform4i(loc, x[0], x[1], x[2], x[3]));
    }

    template <class T>
    void setUniform(const char *name, T const &t) {
        GLuint loc = glGetUniformLocation(impl->id, name);
        if (loc == -1)
            return;
        setUniform(loc, t);
    }
};
