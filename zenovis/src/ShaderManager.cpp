#include <zenovis/ShaderManager.h>
#include <zenovis/opengl/shader.h>
#include <zeno/utils/log.h>

namespace zenovis {

static std::unique_ptr<opengl::Program> _createProgram(
        std::string const &vert,
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

opengl::Program *ShaderManager::compile_program(
        std::string const &vert,
        std::string const &frag,
        std::string const &geo,
        bool supressErr) {
    auto key = vert + frag + geo;
    auto it = _programs.find(key);
    if (it == _programs.end()) {
        std::unique_ptr<opengl::Program> prog;
        if (!supressErr) {
            prog = _createProgram(vert, frag, geo);
        } else {
            try {
                prog = _createProgram(vert, frag, geo);
            } catch (zeno::ErrorException const &e) {
                zeno::log_error("compile_program got error: {}", e.what());
                prog = nullptr;
            }
        }
        auto progPtr = prog.get();
        _programs.emplace(key, std::move(prog));
        return progPtr;
    } else {
        return it->second.get();
    }
}

} // namespace zenovis
