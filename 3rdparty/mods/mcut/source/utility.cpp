#include "mcut/internal/utils.h"


namespace mcut{
#if defined(PROFILING_BUILD)
std::stack<std::unique_ptr<mcut::mini_timer>> g_timestack = std::stack<std::unique_ptr<mcut::mini_timer>>();
#endif
} // namespace mcut{