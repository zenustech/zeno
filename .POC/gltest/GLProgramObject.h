#pragma once

#include "common.h"
#include "GLShaderObject.h"

struct GLProgramObject {
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
                spdlog::error("error linking program: {}", infoLog.data());
            }
        }
        impl = std::make_shared<Impl>();
        impl->id = id;
    }

    void use() {
        CHECK_GL(glUseProgram(impl->id));
    }
};
