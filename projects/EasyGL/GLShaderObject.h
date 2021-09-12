#pragma once

#include "common.h"
#include <zeno/core/IObject.h>

struct GLShaderObject : zeno::IObjectClone<GLShaderObject> {
    struct Impl {
        GLuint id = 0;
        ~Impl() { if (id) glDeleteShader(id); }
    };
    std::shared_ptr<Impl> impl;
    operator GLuint() { return impl->id; }

    void initialize(GLenum type, std::string const &source) {
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
                //zlog::error("error compiling shader: {}", infoLog.data());
            }
        }
        impl = std::make_shared<Impl>();
        impl->id = id;
    }
};
