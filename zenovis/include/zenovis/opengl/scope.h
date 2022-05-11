#pragma once


#include <zenovis/opengl/common.h>
#include <zeno/utils/scope_exit.h>
#include <map>


namespace zenovis::opengl {


inline auto scopeGLEnable(GLenum mode, bool enable = true) {
    return zeno::scope_enter{[&] {
        bool wasEnable = glIsEnabled(mode);
        if (enable)
            CHECK_GL(glEnable(mode));
        else
            CHECK_GL(glDisable(mode));
        return [=] {
            if (wasEnable)
                CHECK_GL(glEnable(mode));
            else
                CHECK_GL(glDisable(mode));
        };
    }};
}

inline auto scopeGLBlendFunc(GLenum src, GLenum dst) {
    return zeno::scope_enter{[&] {
        GLint oldSrc, oldDst;
        CHECK_GL(glGetIntegerv(GL_BLEND_SRC_RGB, &oldSrc));
        CHECK_GL(glGetIntegerv(GL_BLEND_DST_RGB, &oldDst));
        CHECK_GL(glBlendFunc(src, dst));
        return [=] {
            CHECK_GL(glBlendFunc(oldSrc, oldDst));
        };
    }};
}


inline auto scopeGLPixelStorei(GLenum type, GLuint val) {
    return zeno::scope_enter{[&] {
        GLint oldVal;
        CHECK_GL(glGetIntegerv(type, &oldVal));
        CHECK_GL(glPixelStorei(type, val));
        return [=] {
            CHECK_GL(glPixelStorei(type, oldVal));
        };
    }};
}


inline auto scopeGLBindBuffer(GLenum type, GLuint buf) {
    return zeno::scope_enter{[&] {
        GLint oldBuf;
        GLenum bindingType = std::map<GLenum, GLenum>{
            {GL_PIXEL_PACK_BUFFER, GL_PIXEL_PACK_BUFFER_BINDING},
            {GL_PIXEL_UNPACK_BUFFER, GL_PIXEL_UNPACK_BUFFER_BINDING},
        }.at(type);
        CHECK_GL(glGetIntegerv(bindingType, &oldBuf));
        CHECK_GL(glBindBuffer(type, buf));
        return [=] {
            CHECK_GL(glBindBuffer(type, oldBuf));
        };
    }};
}


inline auto scopeGLBindFramebuffer(GLenum type, GLuint fbo) {
    return zeno::scope_enter{[&] {
        GLint oldFbo;
        GLenum bindingType = std::map<GLenum, GLenum>{
            {GL_DRAW_FRAMEBUFFER, GL_DRAW_FRAMEBUFFER_BINDING},
            {GL_READ_FRAMEBUFFER, GL_READ_FRAMEBUFFER_BINDING},
            {GL_FRAMEBUFFER, GL_FRAMEBUFFER_BINDING},
        }.at(type);
        CHECK_GL(glGetIntegerv(bindingType, &oldFbo));
        CHECK_GL(glBindFramebuffer(type, fbo));
        return [=] {
            CHECK_GL(glBindFramebuffer(type, oldFbo));
        };
    }};
}


inline auto scopeGLDrawBuffer(GLuint comp) {
    return zeno::scope_enter{[&] {
        GLint oldComp;
        CHECK_GL(glGetIntegerv(GL_DRAW_BUFFER, &oldComp));
        CHECK_GL(glDrawBuffer(comp));
        return [=] {
            CHECK_GL(glDrawBuffer(oldComp));
        };
    }};
}


inline auto scopeGLReadBuffer(GLuint comp) {
    return zeno::scope_enter{[&] {
        GLint oldComp;
        CHECK_GL(glGetIntegerv(GL_READ_BUFFER, &oldComp));
        CHECK_GL(glReadBuffer(comp));
        return [=] {
            CHECK_GL(glReadBuffer(oldComp));
        };
    }};
}


}
