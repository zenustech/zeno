#pragma once

#include "common.h"

struct GLShaderObject {
    struct Impl {
        GLuint id = 0;
        ~Impl() { if (id) glDeleteShader(id); }
    };
    std::shared_ptr<Impl> impl;
    operator GLuint() { return impl->id; }

    void initialize(std::string const &source, GLenum type) {
        const char *sourcePtr = source.c_str();
        GLint id = glCreateShader(type);
        glShaderSource(id, 1, &sourcePtr, NULL);
        glCompileShader(id);
        GLint status = 0;
        glGetShaderiv(id, GL_COMPILE_STATUS, &status);
        if (!status) {
            GLint infoLen = 0;
            glGetShaderiv(id, GL_INFO_LOG_LENGTH, &infoLen);
            if (infoLen) {
                std::vector<char> infoLog(infoLen);
                glGetShaderInfoLog(id, infoLen, NULL, infoLog.data());
                spdlog::error("error compiling shader: {}", infoLog.data());
            }
        }
        impl->id = id;
    }
};
