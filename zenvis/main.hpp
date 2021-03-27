#pragma once

namespace zenvis {

void set_program_uniforms(Program *pro);

struct IGraphic;

extern std::vector<std::unique_ptr<IGraphic>> graphics;

extern int curr_frameid;

}
