#pragma once

#ifdef ZENO_PARALLEL_STL
#include <execution>
#endif

namespace zeno {

#ifdef ZENO_PARALLEL_STL
#define ZENO_SEQ std::execution::seq,
#define ZENO_PAR std::execution::par,
#define ZENO_PAR_UNSEQ std::execution::par_unseq,
#else
#define ZENO_SEQ /* nothing */
#define ZENO_PAR /* nothing */
#define ZENO_PAR_UNSEQ /* nothing */
#endif

}
