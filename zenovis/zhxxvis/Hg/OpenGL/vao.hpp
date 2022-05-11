#pragma once

namespace hg::OpenGL {

#define NOTHING_CHECK_GL(x) /* nothing */

  struct VAO {
    GLuint vao;

    VAO() { NOTHING_CHECK_GL(glGenVertexArrays(1, &vao)); }

    ~VAO() { NOTHING_CHECK_GL(glDeleteVertexArrays(1, &vao)); }

    void bind() const { NOTHING_CHECK_GL(glBindVertexArray(vao)); }

    void unbind() const { NOTHING_CHECK_GL(glBindVertexArray(0)); }
  };

}  // namespace hg::OpenGL
