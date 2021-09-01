#pragma once

#include "common.h"
#include "GLProgramObject.h"
#include "GLFramebuffer.h"

struct GLVertexAttribInfo {
    GLenum type = GL_FLOAT;
    GLuint dim = 1;
    void *base = nullptr;
    GLuint stride = 0;

    void _bindTo(GLuint index) const {
        glEnableVertexAttribArray(index);
        glVertexAttribPointer(index, dim, type, GL_FALSE, 0, base);
    }
};

static void drawVertexArrays
( GLenum type
, GLuint count
, std::vector<GLVertexAttribInfo> const &vabs
) {
    for (int i = 0; i < vabs.size(); i++) {
        vabs[i]._bindTo(i);
    }
    glDrawArrays(type, 0, count);
    for (int i = 0; i < vabs.size(); i++) {
        glDisableVertexAttribArray(i);
    }
}
