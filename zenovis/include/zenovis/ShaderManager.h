#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <zeno/utils/log.h>
#include <zeno/utils/disable_copy.h>
#include <zenovis/opengl/shader.h>

namespace zenovis {

struct ShaderManager : zeno::disable_copy {
    std::unordered_map<std::string, std::unique_ptr<opengl::Program>> _programs;

    opengl::Program *compile_program(std::string const &vert,
                                     std::string const &frag,
                                     std::string const &geo = {},
                                     bool supressErr = false);
};

} // namespace zenovis
