#pragma once

//#include <glad/glad.h>
#include "../../../include/zenovis/opengl/common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <memory>
#include <string>
#include <vector>

namespace hg::OpenGL {

  static const char *get_opengl_error_string(GLenum err) {
    switch (err) {
#define PER_GL_ERR(x) \
  case x:             \
    return #x;
      PER_GL_ERR(GL_NO_ERROR)
      PER_GL_ERR(GL_INVALID_ENUM)
      PER_GL_ERR(GL_INVALID_VALUE)
      PER_GL_ERR(GL_INVALID_OPERATION)
      PER_GL_ERR(GL_INVALID_FRAMEBUFFER_OPERATION)
      PER_GL_ERR(GL_OUT_OF_MEMORY)
      PER_GL_ERR(GL_STACK_UNDERFLOW)
      PER_GL_ERR(GL_STACK_OVERFLOW)
#undef PER_GL_ERR
    }
    static char tmp[233];
    sprintf(tmp, "%d\n", err);
    return tmp;
  }

  static void _check_opengl_error(const char *file, int line, const char *hint) {
    auto err = glGetError();
    if (err != GL_NO_ERROR) {
      auto msg = get_opengl_error_string(err);
      printf("%s:%d:%s: %s\n", file, line, hint, msg);
      abort();
    }
  }

#define HG_CHECK_GL(x)                                       \
  do {                                                       \
    (x);                                                     \
    hg::OpenGL::_check_opengl_error(__FILE__, __LINE__, #x); \
  } while (0)

}  // namespace hg::OpenGL
