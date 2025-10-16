#pragma once

#include <vector>
#include <string>
#define CUDA_NO_HALF
#include <half.h>

namespace zeno::ChiefDesignerEXR {

int LoadEXR(float **rgba, int *nx, int *ny, const char *filepath, const char **err); 

inline void FreeEXRErrorMessage(const char *err) {
    free((char *)err);
}

int SaveEXR(float *pixels, int width, int height, int channels, int asfp16, const char *filepath, const char **err);

void SaveMultiLayerEXR(std::vector<float*> pixels, int width, int height, std::vector<std::string> channels, const char* exrFilePath); 

void SaveMultiLayerEXR_half(std::vector<Imath::half*> pixels, int width, int height, std::vector<std::string> channels, const char* exrFilePath);

}
