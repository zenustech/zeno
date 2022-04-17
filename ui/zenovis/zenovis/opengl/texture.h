#pragma once

#include <cmath>
#include <stb_image.h>
#include <zeno/utils/disable_copy.h>
#include <zenovis/opengl/common.h>

namespace zenovis::opengl {

struct OpenGLTextureException {};

struct Texture : zeno::disable_copy {
    GLuint tex;
    GLuint target{GL_TEXTURE_2D};
    GLuint wrap_s{GL_CLAMP_TO_EDGE}, wrap_t{GL_CLAMP_TO_EDGE};
    GLuint mag_filter{GL_LINEAR}, min_filter{GL_LINEAR};
    GLuint dtype{GL_UNSIGNED_BYTE};
    GLuint internal_fmt{GL_RGBA8};
    GLuint format{GL_RGBA};

    Texture() {
        CHECK_GL(glGenTextures(1, &tex));
    }

    ~Texture() {
        CHECK_GL(glDeleteTextures(1, &tex));
    }

    void bind_to(int num) const {
        CHECK_GL(glActiveTexture(GL_TEXTURE0 + num));
        CHECK_GL(glBindTexture(target, tex));
    }

    void bind_image(void *img, size_t nx, size_t ny) const {
        CHECK_GL(glBindTexture(target, tex));
        CHECK_GL(glTexParameteri(target, GL_TEXTURE_WRAP_S, wrap_s));
        CHECK_GL(glTexParameteri(target, GL_TEXTURE_WRAP_T, wrap_t));
        CHECK_GL(glTexParameteri(target, GL_TEXTURE_MIN_FILTER, min_filter));
        CHECK_GL(glTexParameteri(target, GL_TEXTURE_MAG_FILTER, mag_filter));
        CHECK_GL(glTexImage2D(target, 0, internal_fmt, nx, ny, 0, format, dtype,
                              img));
        CHECK_GL(glGenerateMipmap(target));
    }

    void load(const char *path) {
        int nx, ny, nc;
        stbi_set_flip_vertically_on_load(true);
        unsigned char *img = stbi_load(path, &nx, &ny, &nc, 0);
        int maxSize;
        glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxSize);
        nx = std::min(std::max(nx, 1), maxSize);
        ny = std::min(std::max(ny, 1), maxSize);
        assert(img);
        switch (nc) {
        case 4: format = GL_RGBA; break;
        case 3: format = GL_RGB; break;
        case 2: format = GL_RG; break;
        case 1: format = GL_RED; break;
        default: throw zeno::makeError("bad texture channel numbers");
        }
        bind_image(img, nx, ny);
        stbi_image_free(img);
    }
};

struct Texture3D : Texture {
    Texture3D() : Texture() {
        target = GL_TEXTURE_3D;
        dtype = GL_FLOAT;
        format = GL_RED;
    }

    void bind_image(void *img, size_t nx, size_t ny) = delete;

    void bind_image(void *img, size_t nx, size_t ny, size_t nz) const {
        CHECK_GL(glBindTexture(target, tex));
        CHECK_GL(glTexParameteri(target, GL_TEXTURE_WRAP_S, wrap_s));
        CHECK_GL(glTexParameteri(target, GL_TEXTURE_WRAP_T, wrap_t));
        CHECK_GL(glTexParameteri(target, GL_TEXTURE_MIN_FILTER, min_filter));
        CHECK_GL(glTexParameteri(target, GL_TEXTURE_MAG_FILTER, mag_filter));
        CHECK_GL(glTexImage3D(target, 0, internal_fmt, nx, ny, nz, 0, format,
                              dtype, img));
    }

    void load(const char *path) = delete;
};

struct FBO : zeno::disable_copy {
    GLuint fbo;
    GLuint target{GL_FRAMEBUFFER};

    FBO() {
        CHECK_GL(glGenFramebuffers(1, &fbo));
    }

    ~FBO() {
        CHECK_GL(glDeleteFramebuffers(1, &fbo));
    }

    void bind() const {
        CHECK_GL(glBindFramebuffer(target, fbo));
    }

    void unbind() const {
        CHECK_GL(glBindFramebuffer(target, 0));
    }

    void bind(GLuint target) const {
        CHECK_GL(glBindFramebuffer(target, fbo));
    }

    void unbind(GLuint target) const {
        CHECK_GL(glBindFramebuffer(target, 0));
    }

    bool complete() const {
        return glCheckFramebufferStatus(target) == GL_FRAMEBUFFER_COMPLETE;
    }

    void attach(Texture const &texture) {
        CHECK_GL(glFramebufferTexture2D(target, GL_COLOR_ATTACHMENT0,
                                        texture.target, texture.tex, 0));
    }
};

} // namespace zenovis::opengl
