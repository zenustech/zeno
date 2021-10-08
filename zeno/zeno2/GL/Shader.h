#pragma once


#include <glm/vec3.hpp>
#include <glm/mat3x3.hpp>
#include <glm/mat4x4.hpp>
#include <zeno2/GL/utils.h>


namespace zeno2::GL {


struct Shader {
    GLuint sha;

    Shader(GLuint type);
    ~Shader();
    void compile(std::string const &source) const;
};


struct Program {
    GLuint pro;

    Program();
    ~Program();
    void attach(Shader const &shader) const;
    void link() const;
    void use() const;
    void set_uniformi(const char *name, int val) const;
    void set_uniform(const char *name, float val) const;
    void set_uniform(const char *name, glm::vec2 const &val) const;
    void set_uniform(const char *name, glm::vec3 const &val) const;
    void set_uniform(const char *name, glm::vec4 const &val) const;
    void set_uniform(const char *name, glm::mat3x3 const &val) const;
    void set_uniform(const char *name, glm::mat4x4 const &val) const;
};

}  // namespace zeno2::GL
