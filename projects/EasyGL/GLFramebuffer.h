#pragma once

#include "common.h"
#include "GLTextureObject.h"
#include <zeno/core/IObject.h>

struct GLFramebuffer : zeno::IObjectClone<GLFramebuffer> {
    struct Impl {
        GLuint id = 0;
        ~Impl() { if (id) glDeleteFramebuffers(1, &id); }
    };
    std::shared_ptr<Impl> impl;
    operator GLuint() { return impl->id; }

    void initialize() {
        impl = std::make_shared<Impl>();
        CHECK_GL(glGenFramebuffers(1, &impl->id));
    }

    void bindToTexture(GLTextureObject const &tex, GLuint attach) const {
        CHECK_GL(glBindFramebuffer(GL_FRAMEBUFFER, impl->id));
        CHECK_GL(glBindTexture(GL_TEXTURE_2D, tex.impl->id));
        CHECK_GL(glFramebufferTexture2D(GL_FRAMEBUFFER, attach,
                    GL_TEXTURE_2D, tex.impl->id, 0));
        CHECK_GL(glBindFramebuffer(GL_FRAMEBUFFER, 0));
    }

    void use() const {
        CHECK_GL(glBindFramebuffer(GL_FRAMEBUFFER, impl ? impl->id : 0));
    }

    void checkStatusComplete() const {
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            //zlog::error("glFramebufferTexture2D: incomplete framebuffer!\n");
        }
    }
};
