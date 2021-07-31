#pragma once

#include "common.h"

struct GLTextureObject {
    struct Impl {
        GLuint id = 0;
        ~Impl() { if (id) glDeleteTextures(1, &id); }
    };
    std::shared_ptr<Impl> impl;
    operator GLuint() { return impl->id; }

    GLuint width = 512;
    GLuint height = 512;
    GLenum format = GL_RGB;
    GLenum type = GL_UNSIGNED_BYTE;
    void *base = nullptr;

    GLenum wrapS = GL_CLAMP_TO_EDGE;
    GLenum wrapT = GL_CLAMP_TO_EDGE;
    GLenum minFilter = GL_LINEAR;
    GLenum magFilter = GL_LINEAR;

    void initialize() {
        impl = std::make_shared<Impl>();
        CHECK_GL(glGenTextures(1, &impl->id));
        CHECK_GL(glBindTexture(GL_TEXTURE_2D, impl->id));
        CHECK_GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrapS));
        CHECK_GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrapT));
        CHECK_GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minFilter));
        CHECK_GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, magFilter));
        CHECK_GL(glTexImage2D(GL_TEXTURE_2D, 0, format, width, height,
                    0, format, type, base));
    }

    void use(GLuint number) const {
        CHECK_GL(glActiveTexture(GL_TEXTURE0 + number));
        CHECK_GL(glBindTexture(GL_TEXTURE_2D, impl->id));
    }
};
