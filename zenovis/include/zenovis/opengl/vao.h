#pragma once

#include <zeno/utils/disable_copy.h>
#include <zenovis/opengl/common.h>

namespace zenovis::opengl {

struct VAO : zeno::disable_copy, ContextBoundResource {
    GLuint vao;

    VAO() {
        CHECK_GL(glGenVertexArrays(1, &vao));
    }

    ~VAO() {
        if (owns_current_context())
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
