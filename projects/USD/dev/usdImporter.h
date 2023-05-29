#ifndef YE_USDIMPORTER_H
#define YE_USDIMPORTER_H

#include "usdDefinition.h"
#include "usdTranslator.h"

#include <pxr/usd/usd/stage.h>

struct EUsdImporter{
    EUsdImporter();

    void OpenStage(EUsdOpenStageOptions options);

    EUsdTranslationContext mTranslationContext;
    pxr::UsdStageRefPtr mStage;
    std::string mIdentifier;
};


#endif //YE_USDIMPORTER_H
