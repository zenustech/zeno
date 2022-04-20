#pragma once

#include <zeno/utils/disable_copy.h>
#include <zenovis/opengl/common.h>

namespace zenovis::opengl {

struct Buffer : zeno::disable_copy {
    GLuint buf;
    GLuint target{GL_ARRAY_BUFFER};

    Buffer(GLuint target = GL_ARRAY_BUFFER) : target(target) {
        CHECK_GL(glGenBuffers(1, &buf));
    }

    ~Buffer() {
        CHECK_GL(glDeleteBuffers(1, &buf));
    }

    template <typename T> void bind_data(std::vector<T> const &arr) const {
        bind_data(arr.data(), arr.size() * sizeof(T));
    }

    template <unsigned N, typename T, glm::qualifier Q>
    void bind_data(std::vector<glm::vec<N, T, Q>> const &arr) const {
        bind_data(arr.data(), arr.size() * N * sizeof(T));
    }

    void bind_data(const void *data, size_t size,
                   GLuint usage = GL_STATIC_DRAW) const {
        CHECK_GL(glBindBuffer(target, buf));
        CHECK_GL(glBufferData(target, size, data, usage));
    }

    void bind_sub_data(const void *data, size_t size, size_t offset) const {
        CHECK_GL(glBufferSubData(target, offset, size, data));
    }

    void attribute(GLuint index, size_t offset, size_t stride, GLuint type,
                   GLuint count) const {
        CHECK_GL(glEnableVertexAttribArray(index));
        CHECK_GL(glVertexAttribPointer(index, count, type, GL_FALSE, stride,
                                       (void *)offset));
    }

    void attribute_p(GLuint index, void *address, size_t stride, GLuint type,
                     GLuint count) const {
        CHECK_GL(glEnableVertexAttribArray(index));
        CHECK_GL(glVertexAttribPointer(index, count, type, GL_FALSE, stride,
                                       address));
    }

    void attrib_divisor(GLuint index, GLuint divisor) const {
        CHECK_GL(glVertexAttribDivisor(index, divisor));
    }

    void disable_attribute(GLuint index) const {
        CHECK_GL(glDisableVertexAttribArray(index));
    }

    void bind() const {
        CHECK_GL(glBindBuffer(target, buf));
    }

    void unbind() const {
        CHECK_GL(glBindBuffer(target, 0));
    }

    void *map(GLuint access = GL_READ_WRITE) const {
        void *ptr;
        CHECK_GL(ptr = glMapBuffer(target, access));
        return ptr;
    }

    void unmap() const {
        CHECK_GL(glUnmapBuffer(target));
    }
};

} // namespace zenovis::opengl
