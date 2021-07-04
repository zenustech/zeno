#pragma once

#include <zeno/zeno.h>
#include "program.h"

struct ProgramObject : zeno::IObjectClone<ProgramObject> {
    Program prog;
};
