#include "usdImporter.h"
#include "usdTranslator.h"

#include <filesystem>

#include <pxr/usd/usd/stageCacheContext.h>
#include <pxr/usd/usdUtils/stageCache.h>
#include <pxr/usd/usd/prim.h>

EUsdImporter::EUsdImporter(){
    mIdentifier = "";
}

void EUsdImporter::OpenStage(EUsdOpenStageOptions options) {
    auto identifier = options.Identifier.c_str();
    ED_COUT << "---------- ---------- ----------\n";
    ED_COUT << "EUsdImporter: OpenStage " << identifier << "\n";

    /// #1 LoadUsdStage
    if(mStage){
        pxr::TfWeakPtr<pxr::SdfLayer> StageRootLayer = mStage->GetRootLayer();
    }

    /// #2 OpenUsdStage
    if (!identifier || std::strlen( identifier ) == 0)
        return;

    pxr::SdfLayerHandleSet LoadedLayers = pxr::SdfLayer::GetLoadedLayers();
    pxr::UsdStageRefPtr Stage;

    // Stage Cache
    if (options.bUseStageCache)
    {
        pxr::UsdStageCacheContext StageCacheContext(pxr::UsdUtilsStageCache::Get());
        auto cacheStackSize = StageCacheContext.GetStack().size();
        ED_COUT << "EUsdImporter: StageCacheContext Size " << cacheStackSize << "\n";
    }

    // Stage Population Mask
    pxr::UsdStagePopulationMask Mask;
    const pxr::VtStringArray* PopulationMask = nullptr;  // Parameter Passed In
    if(PopulationMask)
    {
        // The USD OpenMasked functions don't actually consult or populate the stage cache
        //ensure(options.bUseStageCache == false);
        for (const std::string& AllowedPrimPath : *PopulationMask)
            Mask.Add(pxr::SdfPath{AllowedPrimPath});
    }

    // Stage Load Set
    static_assert((int)pxr::UsdStage::InitialLoadSet::LoadAll == (int)EUsdInitialLoadSet::LoadAll);
    static_assert((int)pxr::UsdStage::InitialLoadSet::LoadNone == (int)EUsdInitialLoadSet::LoadNone);
    pxr::UsdStage::InitialLoadSet LoadSet = static_cast<pxr::UsdStage::InitialLoadSet>(options.InitialLoadSet);

    // Stage Open
    if (std::filesystem::exists(std::filesystem::path(identifier)))
    {
        if(PopulationMask)
            Stage = pxr::UsdStage::OpenMasked(identifier, Mask, LoadSet);
        else
            Stage = pxr::UsdStage::Open(identifier, LoadSet);
    }else{
        ED_CERR << "ERROR: The " << identifier << "doesn't exists.\n";
        return;
    }

    // Force Reload Layers
    if (options.bForceReloadLayersFromDisk && Stage)
    {
        // XXX C:\src\UnrealEngine\Engine\Plugins\Importers\USDImporter\Source\UnrealUSDWrapper\Private\UnrealUSDWrapper.cpp
        // Layers are cached in the layer registry independently of the stage cache. If the layer is already in
        // the registry by the time we try to open a stage, even if we're not using a stage cache at all the
        // layer will be reused and the file will *not* be re-read, so here we manually reload them.
        pxr::SdfLayerHandleVector StageLayers = Stage->GetLayerStack();
        ED_COUT << "EUsdImporter: StageLayers Size " << StageLayers.size() << "\n";
        for (pxr::SdfLayerHandle StageLayer : StageLayers)
        {
            if (LoadedLayers.count(StageLayer) > 0)
                StageLayer->Reload();
        }
    }

    /// #3 Animations
    pxr::SdfLayerHandle rootLayer = Stage->GetRootLayer();
    EUsdLayerTimeInfo layerTimeInfo;
    // TODO Layer Offsets
    layerTimeInfo.Identifier = rootLayer->GetIdentifier();
    layerTimeInfo.FilePath = rootLayer->GetRealPath();
    layerTimeInfo.StartTimeCode = rootLayer->HasStartTimeCode() ? rootLayer->GetStartTimeCode() : 0.0;
    layerTimeInfo.EndTimeCode = rootLayer->HasEndTimeCode() ? rootLayer->GetEndTimeCode() : 0.0;

    ED_COUT << "EUsdImporter: LayerTimeInfo Start " << layerTimeInfo.StartTimeCode << " End " << layerTimeInfo.EndTimeCode << "\n";
    if (pxr::SdfLayerRefPtr Layer = pxr::SdfLayer::FindOrOpen(layerTimeInfo.Identifier)){
        for (const std::string& SubLayerPath : Layer->GetSubLayerPaths()){
            ED_COUT << "EUsdImporter: SubLayerPath " << SubLayerPath << "\n";
            // TODO SubLayer Animation
        }
    }

    /// #3 LoadAssets
    EUsdTranslationContext Context;

    auto startPrim = Stage->GetPrimAtPath(pxr::SdfPath("/"));
    const pxr::TfType ShaderMaterialSchemaType = pxr::TfType::FindByName("UsdShadeMaterial");

    //auto PruneFunc = [](const pxr::UsdPrim& Prim) { ED_COUT << " Prune ("<<Prim.GetPath()<<")\n"; return false;};
    FunctionPrune PruneFunc = [&Context](const pxr::UsdPrim& UsdPrim) -> bool
    {
        auto Translator = Context.GetTranslatorByPrim(UsdPrim);
        if(Translator != nullptr){
            EUsdTranslatorContext translatorContext{UsdPrim, {}};
            bool Collapses = Translator->CollapsesChildren(translatorContext, ECollapsingType::Assets);
            ED_COUT << " Prune ("<<UsdPrim.GetPath()<<") " << Collapses << "\n";
            return Collapses;
        }

        return false;
    };

    ED_COUT << " --- Get Materials Prims Start ---\n";
    pxr::VtArray<pxr::UsdPrim> AllMaterialAssets = GetAllPrimsOfType(startPrim, ShaderMaterialSchemaType, PruneFunc);
    ED_COUT << " --- Get Materials Prims End ---\n";

    ED_COUT << "CreateMaterials Size " << AllMaterialAssets.size() << "\n";

    auto CreateAssets = [&Context, this] (const pxr::UsdPrim& Prim, EConvertExtraInfo ExtraInfo) -> bool
    {
        auto Translator = Context.GetTranslatorByPrim(Prim);
        if(Translator != nullptr) {
            EUsdTranslatorContext translatorContext{Prim, ExtraInfo};

            ED_COUT << "========== " << Prim.GetPath() << " ========== Type " << Prim.GetTypeName() << "\n";
            return Translator->Create(translatorContext);
        }

        return false;
    };

    auto CreateAssetsForPrims = [&Stage, &CreateAssets](const pxr::VtArray<pxr::UsdPrim>& AllPrimAssets)
    {
        // Create Ignores
        pxr::VtStringArray ConvertMeshIgnores{};
        for (const pxr::UsdPrim& UsdPrim : AllPrimAssets){
            auto Path = UsdPrim.GetPath();
            auto Type = UsdPrim.GetTypeName();
            if(Type == "SkelRoot"){
                ConvertMeshIgnores.emplace_back(Path.GetAsString());
            }
        }

        EConvertExtraInfo ExtraInfo;
        ExtraInfo.ConvertMeshIgnores = ConvertMeshIgnores;

        for (const pxr::UsdPrim& UsdPrim : AllPrimAssets)
        {
            auto Path = UsdPrim.GetPath();
            auto Type = UsdPrim.GetTypeName();
            //ED_COUT << " CreateAsset " << Path << " Type " << Type << "\n";
            CreateAssets(UsdPrim, ExtraInfo);
        }
    };

    ED_COUT << "CreatePrims Size For Materials " << AllMaterialAssets.size() << "\n";
    for(auto const& MatAssets: AllMaterialAssets){
        ED_COUT << " - " << MatAssets.GetPath() << "\n";
    }
    CreateAssetsForPrims(AllMaterialAssets);

    FunctionPrune UsdPruneChildren = [&PruneFunc](const pxr::UsdPrim& ChildPrim) -> bool
    {
        ED_COUT << "       - Usd Prune Children " << ChildPrim.GetPath() << "\n";
        return PruneFunc(pxr::UsdPrim(ChildPrim));
    };

    const pxr::TfType BaseSchemaType = pxr::TfType::FindByName("UsdSchemaBase");

    ED_COUT << " --- Get Base Prims Start ---\n";
    pxr::VtArray<pxr::UsdPrim> AllOthersAssets = GetAllPrimsOfType(startPrim, BaseSchemaType, UsdPruneChildren, {ShaderMaterialSchemaType});
    ED_COUT << " --- Get Base Prims End ---\n";

    ED_COUT << "CreatePrims Size For Prims " << AllOthersAssets.size() << "\n";
    for(auto const& PrimAssets: AllOthersAssets){
        ED_COUT << " - " << PrimAssets.GetPath() << "\n";
    }
    CreateAssetsForPrims(AllOthersAssets);

    mStage = Stage;
    mIdentifier = options.Identifier;
    mTranslationContext = Context;
    ED_COUT << "---------- ---------- ----------\n";
}
