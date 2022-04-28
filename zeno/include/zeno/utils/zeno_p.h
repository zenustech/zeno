#pragma once

#include <zeno/utils/log.h>
#include <zeno/utils/glm_to_string.h>

#define ZENO_P(x) ::zeno::log_critical(#x " = {}", (x))
