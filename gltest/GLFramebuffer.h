#pragma once

#include "common.h"

struct GLFramebuffer {
    struct Impl {
        GLuint id = 0;
        ~Impl() { if (id) glDeleteFramebuffers(1, &id); }
    };
    std::shared_ptr<Impl> impl;
    operator GLuint() { return impl->id; }

    GLuint width = 512;
    GLuint height = 512;
    GLenum format = GL_RGB;
    GLenum type = GL_UNSIGNED_BYTE;

    GLenum wrapS = GL_CLAMP_TO_EDGE;
    GLenum wrapT = GL_CLAMP_TO_EDGE;
    GLenum minFilter = GL_LINEAR;
    GLenum magFilter = GL_LINEAR;

    void initialize() {
        CHECK_GL(glGenFramebuffers(1, &impl->id));
    }

    void _bindToTexture(GLTexture const &tex, GLuint attach = GL_COLOR_ATTACHMENT0) {
        CHECK_GL(glBindFramebuffer(GL_FRAMEBUFFER, impl->id));
        CHECK_GL(glBindTexture(GL_TEXTURE_2D, tex.impl->id));
        CHECK_GL(glFrambufferTexture2D(GL_FRAMEBUFFER, attach,
                    GL_TEXTURE_2D, tex.impl->id, 0));
    }

    void _checkStatusComplete() {
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            spdlog::error("glFramebufferTexture2D: incomplete framebuffer!\n");
        }
    }
};
