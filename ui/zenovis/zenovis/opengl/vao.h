#pragma once

#include <zeno/utils/disable_copy.h>
#include <zenovis/opengl/common.h>

namespace zenovis::opengl {

struct VAO : zeno::disable_copy {
    GLuint vao;

    VAO() {
        CHECK_GL(glGenVertexArrays(1, &vao));
    }

    ~VAO() {
        CHECK_GL(glDeleteVertexArrays(1, &vao));
    }

    void bind() const {
        CHECK_GL(glBindVertexArray(vao));
    }

    void unbind() const {
        CHECK_GL(glBindVertexArray(0));
    }
};

} // namespace zenovis::opengl
