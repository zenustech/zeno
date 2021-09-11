#pragma once

#include <memory>
#include <cstdio>
#include <vector>
#include <string>
#include <cstdlib>
#include <cstring>
#include <GLES3/gl3.h>
//#include <zeno/utils/zlog.h>

static const char *get_opengl_error_string(GLenum err) {
    switch (err) {
#define PER_GL_ERR(x) \
        case x:  return #x;
        PER_GL_ERR(GL_NO_ERROR)
        PER_GL_ERR(GL_INVALID_ENUM)
        PER_GL_ERR(GL_INVALID_VALUE)
        PER_GL_ERR(GL_INVALID_OPERATION)
        PER_GL_ERR(GL_INVALID_FRAMEBUFFER_OPERATION)
        PER_GL_ERR(GL_OUT_OF_MEMORY)
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
        //zlog::error("{}:{}: `{}`: {}", file, line, hint, msg);
    }
}

#define CHECK_GL(x) do { (x); \
    _check_opengl_error(__FILE__, __LINE__, #x); \
  } while (0)
