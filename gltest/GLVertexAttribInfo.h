#pragma once

#include "common.h"

struct GLVertexAttribInfo {
    GLenum type = GL_FLOAT;
    GLuint dim = 1;
    void *base = nullptr;
    GLuint stride = 0;

    void bindTo(GLuint index) {
        glEnableVertexAttribArray(index);
        glVertexAttribPointer(index, dim, type, GL_FALSE, 0, base);
    }
};
