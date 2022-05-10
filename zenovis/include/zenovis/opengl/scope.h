#pragma once


#include <zenovis/opengl/common.h>
#include <zeno/utils/scope_exit.h>
#include <optional>


namespace zenovis::opengl {


inline auto scopeGLEnable(GLenum mode, bool enable = true) {
    return zeno::scope_enter{[&] {
        bool wasEnable = glIsEnabled(mode);
        if (enable)
            CHECK_GL(glEnable(mode));
        else
            CHECK_GL(glDisable(mode));
        printf("outer %d\n", mode);
        return [=] {
            printf("inner %d\n", mode);
            if (wasEnable)
                CHECK_GL(glEnable(mode));
            else
                CHECK_GL(glDisable(mode));
        };
    }};
}

inline auto scopeGLBlendFunc(GLenum arg1, GLenum arg2) {
    return zeno::scope_enter{[&] {
        GLint oldArg1, oldArg2;
        CHECK_GL(glGetIntegerv(GL_BLEND_SRC_RGB, &oldArg1));
        CHECK_GL(glGetIntegerv(GL_BLEND_DST_RGB, &oldArg2));
        CHECK_GL(glBlendFunc(arg1, arg2));
        return [=] {
            CHECK_GL(glBlendFunc(oldArg1, oldArg2));
        };
    }};
}


}
