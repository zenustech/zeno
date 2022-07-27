#include "zensim/cuda/Cuda.h"

namespace zeno {
struct ZpcInitializer {
  ZpcInitializer() {
    printf("Initializing ZPC...\n");
    (void)zs::Cuda::instance();
    printf("Initialized ZPC successfully!\n");
  }
};
static ZpcInitializer g_zpc_initializer{};
} // namespace zeno
