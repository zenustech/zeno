#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <zeno/utils/disable_copy.h>
#include <zenovis/opengl/shader.h>

namespace zenovis {

struct ShaderManager : zeno::disable_copy {
    std::unique_ptr<opengl::Program> _createProgram(std::string const &vert,
                                                    std::string const &frag,
                                                    std::string const &geo) {
        auto pro = std::make_unique<opengl::Program>();
        auto vs = std::make_unique<opengl::Shader>(GL_VERTEX_SHADER);
        auto fs = std::make_unique<opengl::Shader>(GL_FRAGMENT_SHADER);
        std::unique_ptr<opengl::Shader> gs;
        if (geo.size())
            gs = std::make_unique<opengl::Shader>(GL_GEOMETRY_SHADER);
        vs->compile(vert);
        fs->compile(frag);

        pro->attach(*vs);
        pro->attach(*fs);
        if (geo.size()) {
            gs->compile(geo);
            pro->attach(*gs);
        }
        pro->link();
        return pro;
    }

    std::unordered_map<std::string, std::unique_ptr<opengl::Program>> _programs;

    opengl::Program *compile_program(std::string const &vert,
                                     std::string const &frag,
                                     std::string const &geo = {}) {
        auto key = vert + frag + geo;
        auto it = _programs.find(key);
        if (it == _programs.end()) {
            auto prog = _createProgram(vert, frag, geo);
            auto progPtr = prog.get();
            _programs.emplace(key, std::move(prog));
            return progPtr;
        } else {
            return it->second.get();
        }
    }
};

} // namespace zenovis
