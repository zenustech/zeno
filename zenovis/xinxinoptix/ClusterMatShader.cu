// A separate SBT record keeps ordinary node-graph geometry on its general
// shader. CLAS can fold unsupported material features out of the same BSDF.
#define ZENO_CLAS_ONLY 1
#include "DeflMatShader.cu"
