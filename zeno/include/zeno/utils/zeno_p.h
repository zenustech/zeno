#pragma once

#include <zeno/utils/glm_to_string.h>
#include <zeno/utils/log.h>

#ifndef ZENO_NOTICKTOCK
#define ZENO_P(x) ::zeno::log_critical(#x " = {}", (x))
#endif
