#pragma once

#include "stdafx.hpp"

namespace zenvis {

Program *compile_program(std::string const &vert, std::string const &frag, std::string const &geo="");

}
