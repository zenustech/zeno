#pragma once

namespace hg::OpenGL {

  struct VAO {
    GLuint vao;

    VAO() { CHECK_GL(glGenVertexArrays(1, &vao)); }

    ~VAO() { CHECK_GL(glDeleteVertexArrays(1, &vao)); }

    void bind() const { CHECK_GL(glBindVertexArray(vao)); }

    void unbind() const { CHECK_GL(glBindVertexArray(0)); }
  };

}  // namespace hg::OpenGL
