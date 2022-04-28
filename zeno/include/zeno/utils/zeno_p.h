#pragma once

#include <zeno/utils/glm_to_string.h>
#include <zeno/utils/log.h>

#define ZENO_P(x) ::zeno::log_critical(#x " = {}", (x))
