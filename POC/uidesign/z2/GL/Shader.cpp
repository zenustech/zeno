#include <z2/GL/Shader.h>
#include <glm/gtc/type_ptr.hpp>


namespace z2::GL {


Shader::Shader(GLuint type) {
    CHECK_GL(sha = glCreateShader(type));
}


Shader::~Shader() {
    CHECK_GL(glDeleteShader(sha));
}


void Shader::compile(std::string const &source) const {
    const GLchar *src = source.c_str();
    CHECK_GL(glShaderSource(sha, 1, &src, nullptr));
    CHECK_GL(glCompileShader(sha));
    int status = GL_TRUE;
    CHECK_GL(glGetShaderiv(sha, GL_COMPILE_STATUS, &status));
    if (status != GL_TRUE) {
        GLsizei logLength;
        CHECK_GL(glGetShaderiv(sha, GL_INFO_LOG_LENGTH, &logLength));
        std::vector<GLchar> log(logLength + 1);
        CHECK_GL(glGetShaderInfoLog(sha, logLength, &logLength, log.data()));
        log[logLength] = 0;
        throw ztd::format_error("Error compiling shader:\n%s\n%s\n",
                                source.c_str(), log.data());
    }
}


Program::Program() {
    CHECK_GL(pro = glCreateProgram());
}

Program::~Program() {
    CHECK_GL(glDeleteProgram(pro));
}

void Program::attach(Shader const &shader) const {
    CHECK_GL(glAttachShader(pro, shader.sha));
}

void Program::link() const {
    CHECK_GL(glLinkProgram(pro));
    int status = GL_TRUE;
    CHECK_GL(glGetProgramiv(pro, GL_LINK_STATUS, &status));
    if (status != GL_TRUE) {
        GLsizei logLength;
        CHECK_GL(glGetProgramiv(pro, GL_INFO_LOG_LENGTH, &status));
        std::vector<GLchar> log(logLength + 1);
        CHECK_GL(glGetProgramInfoLog(pro, logLength, &logLength, log.data()));
        log[logLength] = 0;
        throw ztd::format_error("Error linking program:\n%s\n", log.data());
    }
}

void Program::use() const {
    CHECK_GL(glUseProgram(pro));
}

void Program::set_uniformi(const char *name, int val) const {
    GLuint loc;
    CHECK_GL(loc = glGetUniformLocation(pro, name));
    if (loc == -1) return;
    CHECK_GL(glUniform1i(loc, val));
}

void Program::set_uniform(const char *name, float val) const {
    GLuint loc;
    CHECK_GL(loc = glGetUniformLocation(pro, name));
    if (loc == -1) return;
    CHECK_GL(glUniform1f(loc, val));
}

void Program::set_uniform(const char *name, glm::vec2 const &val) const {
    GLuint loc;
    CHECK_GL(loc = glGetUniformLocation(pro, name));
    if (loc == -1) return;
    CHECK_GL(glUniform2fv(loc, 1, glm::value_ptr(val)));
}

void Program::set_uniform(const char *name, glm::vec3 const &val) const {
    GLuint loc;
    CHECK_GL(loc = glGetUniformLocation(pro, name));
    if (loc == -1) return;
    CHECK_GL(glUniform3fv(loc, 1, glm::value_ptr(val)));
}

void Program::set_uniform(const char *name, glm::vec4 const &val) const {
    GLuint loc;
    CHECK_GL(loc = glGetUniformLocation(pro, name));
    if (loc == -1) return;
    CHECK_GL(glUniform4fv(loc, 1, glm::value_ptr(val)));
}

void Program::set_uniform(const char *name, glm::mat3x3 const &val) const {
    GLuint loc;
    CHECK_GL(loc = glGetUniformLocation(pro, name));
    if (loc == -1) return;
    CHECK_GL(glUniformMatrix3fv(loc, 1, GL_FALSE, glm::value_ptr(val)));
}

void Program::set_uniform(const char *name, glm::mat4x4 const &val) const {
    GLuint loc;
    CHECK_GL(loc = glGetUniformLocation(pro, name));
    if (loc == -1) return;
    CHECK_GL(glUniformMatrix4fv(loc, 1, GL_FALSE, glm::value_ptr(val)));
}

//void Program::set_uniform(const char *name, glm::mat4x3 const &val) const {
    //GLuint loc;
    //CHECK_GL(loc = glGetUniformLocation(pro, name));
    //CHECK_GL(glUniformMatrix4x3fv(loc, 1, GL_FALSE, glm::value_ptr(val)));
//}

//void Program::set_uniform(const char *name, glm::mat3x4 const &val) const {
    //GLuint loc;
    //CHECK_GL(loc = glGetUniformLocation(pro, name));
    //CHECK_GL(glUniformMatrix3x4fv(loc, 1, GL_FALSE, glm::value_ptr(val)));
//}

}
