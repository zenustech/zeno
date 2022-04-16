#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <zeno/utils/disable_copy.h>
#include <zenovis/opengl/shader.h>

namespace zenovis {

struct ShaderManager : zeno::disable_copy {
    std::unordered_map<std::string, std::unique_ptr<opengl::Program>> _programs;

    struct _ManagedProgram : opengl::Program {
        std::unique_ptr<opengl::Shader> vs, fs, gs;

        _ManagedProgram(std::string const &vert, std::string const &frag,
                  std::string const &geo) {
            vs = std::make_unique<opengl::Shader>(GL_VERTEX_SHADER);
            fs = std::make_unique<opengl::Shader>(GL_FRAGMENT_SHADER);
            if (geo.size())
                gs = std::make_unique<opengl::Shader>(GL_GEOMETRY_SHADER);
            vs->compile(vert);
            fs->compile(frag);

            attach(*vs);
            attach(*fs);
            if (geo.size()) {
                gs->compile(geo);
                attach(*gs);
            }
            link();
        }
    };

    opengl::Program *compile_program(std::string const &vert,
                                     std::string const &frag,
                                     std::string const &geo = {}) {
        auto key = vert + frag + geo;
        auto it = _programs.find(key);
        if (it == _programs.end()) {
            auto prog = std::make_unique<_ManagedProgram>(vert, frag, geo);
            auto progPtr = prog.get();
            _programs.emplace(key, std::move(prog));
            return progPtr;
        } else {
            return it->second.get();
        }
    }
};

} // namespace zenovis
