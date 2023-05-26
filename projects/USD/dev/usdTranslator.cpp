#include "usdTranslator.h"

#include <iostream>

#include "pxr/usd/usdGeom/mesh.h"
#include "pxr/usd/usdShade/material.h"
#include "pxr/usd/usdSkel/binding.h"
#include "pxr/usd/usdSkel/bindingAPI.h"
#include "pxr/usd/usdSkel/blendShapeQuery.h"
#include "pxr/usd/usdSkel/cache.h"
#include "pxr/usd/usdSkel/root.h"
#include "pxr/usd/usdSkel/skeletonQuery.h"
#include "pxr/usd/usdSkel/tokens.h"

#include <pxr/usd/usdShade/shader.h>
#include <pxr/usd/usdShade/material.h>
#include <pxr/usd/usd/modelAPI.h>

std::map<std::string, int32>& EUsdTranslationContext::FindOrAddPrimvarToUVIndex(std::string primvar){
    if(MaterialToPrimvarToUVIndex.find(primvar) == MaterialToPrimvarToUVIndex.end()){
        std::map<std::string, int32> Unused;
        MaterialToPrimvarToUVIndex.emplace(primvar, Unused);
    }
    return MaterialToPrimvarToUVIndex[primvar];
}

EUsdTranslationContext::EUsdTranslationContext() {
    // The order of translator will affect the result of translation.
    //  1. We move `UsdSkelRoot` to the first, instead of using `UsdGeomXformable` to convert it.
    auto& RST = RegisteredSchemaTranslators;
    RST.emplace_back("UsdSkelRoot",                 ESchemaTranslatorContext{std::make_shared<EUsdSkelRootTranslator>(),            pxr::TfToken("")});
    RST.emplace_back("UsdGeomCamera",               ESchemaTranslatorContext{std::make_shared<EUsdGeomCameraTranslator>(),          pxr::TfToken("")});
    RST.emplace_back("UsdGeomMesh",                 ESchemaTranslatorContext{std::make_shared<EUsdGeomMeshTranslator>(),            pxr::TfToken("")});
    RST.emplace_back("UsdGeomPointInstancer",       ESchemaTranslatorContext{std::make_shared<EUsdGeomPointInstancerTranslator>(),  pxr::TfToken("")});
    RST.emplace_back("UsdGeomXformable",            ESchemaTranslatorContext{std::make_shared<EUsdGeomXformableTranslator>(),       pxr::TfToken("")});
    RST.emplace_back("UsdShadeMaterial",            ESchemaTranslatorContext{std::make_shared<EUsdShadeMaterialTranslator>(),       pxr::TfToken("")});
    RST.emplace_back("UsdLuxBoundableLightBase",    ESchemaTranslatorContext{std::make_shared<EUsdLuxLightTranslator>(),            pxr::TfToken("")});
    RST.emplace_back("UsdLuxNonboundableLightBase", ESchemaTranslatorContext{std::make_shared<EUsdLuxLightTranslator>(),            pxr::TfToken("")});
    RST.emplace_back("UsdGeomXformable",            ESchemaTranslatorContext{std::make_shared<EUsdGroomTranslator>(),               pxr::TfToken("")});
    for(auto& T:RST) {
        T.second.Translator->Context = this;
    }

    auto& RPT = RegisteredPrimTypeTranslators;
    RPT.emplace_back("BasisCurves",                 EPrimTypeTranslatorContext{std::make_shared<EBasisCurvesTranslator>()});
    RPT.emplace_back("Points",                      EPrimTypeTranslatorContext{std::make_shared<EPointsTranslator>()});

    for(auto& T:RPT) {
        T.second.Translator->Context = this;
    }

    //std::reverse(RST.begin(), RST.end());
}

std::shared_ptr<EUsdSchemaTranslator> EUsdTranslationContext::GetTranslatorByName(std::string Name) {
    auto& RST = RegisteredSchemaTranslators;
    for(auto& T:RST){
        if(T.first == Name){
            return T.second.Translator;
        }
    }
    ED_CERR << "ERROR: No Translator For " << Name << "\n";
    return nullptr;
}

std::shared_ptr<EUsdSchemaTranslator> EUsdTranslationContext::GetTranslatorByPrim(const pxr::UsdPrim& Prim){
    for(auto& schemaTranslator: RegisteredSchemaTranslators){
        pxr::TfToken RegisteredSchemaToken(schemaTranslator.first);
        pxr::TfType RegisteredSchemaType = pxr::UsdSchemaRegistry::GetTypeFromName(RegisteredSchemaToken);
        auto isSchemaType = Prim.IsA(RegisteredSchemaType);
        if(isSchemaType){
            ED_COUT << " Get Translator " << schemaTranslator.first << "\n";
            return schemaTranslator.second.Translator;
        }
    }
    //ED_CERR << "ERROR: No Translator For " << Prim.GetPath() << "\n";
    return nullptr;
}

bool EUsdSchemaTranslator::Create(EUsdTranslatorContext& context) {
    ED_CERR << "ERROR: Called Base CreateFunction\n";
    return false;
}

bool EPrimTypeTranslator::Create(EUsdTranslatorContext &context) {
    ED_CERR << "ERROR: Called Base CreateFunction\n";
    return false;
}

bool EUsdShadeMaterialTranslator::Create(EUsdTranslatorContext& context) {
    // Functions

    pxr::UsdShadeMaterial ShadeMaterial(context.Prim);
    if (!ShadeMaterial)
        return false;

    const pxr::TfToken RenderContextToken = pxr::UsdShadeTokens->universalRenderContext;
    auto PrimPath = context.Prim.GetPrimPath();

    // TODO Hash Material for Cache
    //HashShadeMaterial(ShadeMaterial, RenderContextToken);

    /// Step1: ConvertMaterial
    std::map<std::string, int32> Unused;
    auto key = PrimPath.GetAsString();
    std::map<std::string, int32>& PrimvarToUVIndex = Context->FindOrAddPrimvarToUVIndex(key);

    const bool bSuccess = ConvertMaterial(ShadeMaterial, PrimvarToUVIndex);

    if (!bSuccess)
        return false;
}

bool EUsdSkelRootTranslator::Create(EUsdTranslatorContext &context) {
    auto PrimPath = context.Prim.GetPath();
    ED_COUT << "SkelRoot Translator Create: " << PrimPath << "\n";
    // SetupTasks

    /// Task: LoadAllSkeletalData
    LoadAllSkeletalData(context);

    /// Task: Create AnimSequences
    CreateSkeletalAnim(context);

    ESkelImportData ImportData;
    ImportData.SkeletonBones = SkeletonBones;
    ImportData.BlendShapes = NewBlendShapes;
    ImportData.SkeletalMeshData = PathToSkeletalMeshImportData;
    ImportData.SkeletalAnimData = SkeletalAnimData;

    Context->Imported.PathToSkelImportData[PrimPath.GetAsString()] = ImportData;

    SkeletonBones.clear();
    SkeletonName = {};
    NewBlendShapes.clear();
    UsedMorphTargetNames.clear();
    SkeletonCache = {};
    PathToSkeletalMeshImportData.clear();
    SkeletalAnimData = {};

    return true;
}

bool EUsdSkelRootTranslator::CreateSkeletalAnim(EUsdTranslatorContext &context){
    if (pxr::UsdSkelRoot SkeletonRoot{context.Prim}){
        std::vector<pxr::UsdSkelBinding> SkeletonBindings;
        SkeletonCache.Populate(SkeletonRoot, pxr::UsdTraverseInstanceProxies());
        SkeletonCache.ComputeSkelBindings(SkeletonRoot, &SkeletonBindings, pxr::UsdTraverseInstanceProxies());

        bool HasAnimated = false;
        for (const pxr::UsdSkelBinding& Binding : SkeletonBindings){
            const pxr::UsdSkelSkeleton& Skeleton = Binding.GetSkeleton();
            pxr::UsdSkelSkeletonQuery SkelQuery = SkeletonCache.GetSkelQuery(Skeleton);
            pxr::UsdSkelAnimQuery AnimQuery = SkelQuery.GetAnimQuery();
            if (!AnimQuery) {
                continue;
            }

            pxr::UsdPrim SkelAnimationPrim = AnimQuery.GetPrim();
            if (!SkelAnimationPrim) {
                continue;
            }

            std::string SkelAnimationPrimPath = SkelAnimationPrim.GetPath().GetAsString();

            std::vector<double> JointTimeSamples;
            std::vector<double> BlendShapeTimeSamples;
            if ((!AnimQuery.GetJointTransformTimeSamples(&JointTimeSamples) || JointTimeSamples.size() == 0) &&
                 (NewBlendShapes.size() == 0 || (!AnimQuery.GetBlendShapeWeightTimeSamples(&BlendShapeTimeSamples)
                 || BlendShapeTimeSamples.size() == 0)))
            {
                ED_COUT << " Continue Samples " << SkelAnimationPrimPath << "\n";
                continue;
            }

            // TODO RootMotionHandling
            pxr::UsdPrim RootMotionPrim = SkeletonRoot.GetPrim();

            // ... AnimSequence
            if(true){
                HasAnimated = true;
                pxr::VtArray<pxr::UsdSkelSkinningQuery> SkinningTargets = Binding.GetSkinningTargets();
                float LayerStartOffsetSeconds = 0.0f;
                ConvertSkelAnim(SkelQuery, &SkinningTargets, SkeletonBones,
                                &NewBlendShapes, SkeletalAnimData, RootMotionPrim, &LayerStartOffsetSeconds);
            }
            break;
        }
        if(! HasAnimated){
            ETransform RootMotionTransform;
            pxr::UsdStageWeakPtr _Stage = SkeletonRoot.GetPrim().GetStage();
            pxr::UsdGeomXformable _Xformable = pxr::UsdGeomXformable{SkeletonRoot.GetPrim()};
            ConvertXformable(
                    _Stage,
                    _Xformable,
                    RootMotionTransform,
                    pxr::UsdTimeCode::EarliestTime().GetValue()
            );
            ED_COUT << "Not Animated, Get Root Transform: " << RootMotionTransform << "\n";
            SkeletalAnimData.RootMotionTransforms.emplace_back(RootMotionTransform);
        }
    }
}

bool EUsdSkelRootTranslator::LoadAllSkeletalData(EUsdTranslatorContext &context) {
    pxr::UsdSkelCache& InSkeletonCache = SkeletonCache;
    pxr::UsdSkelRoot InSkeletonRoot = pxr::UsdSkelRoot(context.Prim);
    auto& OutSkeletonBones = SkeletonBones;
    auto& OutSkeletonName = SkeletonName;
    auto& OutBlendShapes = NewBlendShapes;
    auto& InMaterialToPrimvarsUVSetNames = Context->MaterialToPrimvarToUVIndex;
    auto& InOutUsedMorphTargetNames = UsedMorphTargetNames;
    auto& OutSkeletalTransforms = SkeletalAnimData;
    auto InTime = 0.0f;
    auto& OutPathToSkeletalMeshImportData = PathToSkeletalMeshImportData;

    if (!InSkeletonRoot) {
        ED_CERR << "ERROR: Invalid Skeleton Root\n";
        return false;
    }

    pxr::UsdStageRefPtr Stage = context.Prim.GetStage();
    const EUsdStageInfo StageInfo(Stage);

    std::vector<pxr::UsdSkelBinding> SkeletonBindings;
    InSkeletonCache.Populate(InSkeletonRoot, pxr::UsdTraverseInstanceProxies());
    InSkeletonCache.ComputeSkelBindings(InSkeletonRoot, &SkeletonBindings, pxr::UsdTraverseInstanceProxies());
    ED_COUT << "   Skel: SkeletonBindings " << SkeletonBindings.size() << "\n";
    if (SkeletonBindings.size() < 1)
    {
        ED_CERR << "ERROR: InvalidBinding, SkelRoot " << InSkeletonRoot.GetPath() << " doesn't have any binding. No skinned mesh will be generated.\n";
        return false;
    }

    // Note that there could be multiple skeleton bindings under the SkeletonRoot, For now, extract just the first one
    const pxr::UsdSkelBinding& SkeletonBinding = SkeletonBindings[0];
    const pxr::UsdSkelSkeleton& Skeleton = SkeletonBinding.GetSkeleton();
    if (SkeletonBindings.size() > 1)
    {
        ED_COUT << "Currently only a single skeleton is supported per UsdSkelRoot! '"  << InSkeletonRoot.GetPrim().GetPath() << "' will use skeleton '" << Skeleton.GetPrim().GetPath() << "'\n";
    }
    // Import skeleton data
    pxr::UsdSkelSkeletonQuery SkelQuery = InSkeletonCache.GetSkelQuery(Skeleton);
    {
        ESkeletalMeshImportData DummyImportData;
        const bool bSkeletonValid = ConvertSkeleton(SkelQuery, DummyImportData);
        if (!bSkeletonValid)
        {
            ED_CERR << "ERROR: Invalid Skeleton for Convert Skeleton\n";
            return false;
        }
        OutSkeletonBones = DummyImportData.RefBonesBinary;
        OutSkeletonName = SkelQuery.GetSkeleton().GetPrim().GetName();

        ED_COUT << "   Skel: OutSkeletonBones " << OutSkeletonBones.size() << "\n";
        ED_COUT << "   Skel: OutSkeletonName " << OutSkeletonName << "\n";
    }

    //std::map<int32, ESkeletalMeshImportData> SkeletalMeshImportDataMap;
    std::map<std::string, ESkeletalMeshImportData> SkeletalMeshImportDataMap;

    // Since we may need to switch variants to parse, we could invalidate references to SkinningQuery objects,
    // so we need to keep track of these by path and construct one whenever we need them
    pxr::VtArray<pxr::SdfPath> PathsToSkinnedPrims;
    for (const pxr::UsdSkelSkinningQuery& SkinningQuery : SkeletonBinding.GetSkinningTargets())
    {
        // In USD, the skinning target need not be a mesh
        if (pxr::UsdGeomMesh SkinningMesh = pxr::UsdGeomMesh(SkinningQuery.GetPrim()))
            PathsToSkinnedPrims.emplace_back(SkinningMesh.GetPrim().GetPath());
    }
    //ED_COUT << "   Skel: PathsToSkinnedPrims " << PathsToSkinnedPrims << "\n";

    auto ConvertSkinnedPrim = [
            &SkeletalMeshImportDataMap,
            &SkelQuery,
            InTime,
            &InMaterialToPrimvarsUVSetNames,
            &InOutUsedMorphTargetNames,
            &OutBlendShapes,
            &StageInfo,
            &OutSkeletalTransforms
    ](const pxr::UsdGeomMesh& GeomMesh) -> bool
    {
        pxr::UsdSkelSkinningQuery SkinningQuery = CreateSkinningQuery(GeomMesh, SkelQuery);
        if (!SkinningQuery)
            return true;
        if (GeomMesh && GeomMesh.ComputeVisibility() == pxr::UsdGeomTokens->invisible)
            return true;
        ED_COUT << "Convert Skel Mesh: " << GeomMesh.GetPath() << "\n";

        ESkeletalMeshImportData& ImportData = SkeletalMeshImportDataMap[GeomMesh.GetPath().GetAsString()];

        // BlendShape data is respective to point indices for each mesh in isolation, combine all points
        //  into one?
        uint32 NumPointsBeforeThisMesh = static_cast<uint32>(ImportData.MeshData.Points.size());
        ED_COUT << " = NumPointsBeforeThisMesh " << NumPointsBeforeThisMesh << "\n";

        bool bSuccess = ConvertSkinnedMesh(
                SkinningQuery,
                SkelQuery,
                ImportData,
                InMaterialToPrimvarsUVSetNames,
                pxr::UsdShadeTokens->universalRenderContext,
                pxr::UsdShadeTokens->allPurpose,
                OutSkeletalTransforms
        );
        if (!bSuccess)
            return true;

        // BlendShapes
        if(true){
            pxr::UsdSkelBindingAPI SkelBindingAPI{GeomMesh.GetPrim()};
            pxr::UsdSkelBlendShapeQuery BlendShapeQuery{SkelBindingAPI};

            if (BlendShapeQuery){
                ED_COUT << " Geom Mesh Has BlendShape " << GeomMesh.GetPath() << " Size " << BlendShapeQuery.GetNumBlendShapes() << "\n";

                pxr::UsdAttribute BlendShapesAttr = SkelBindingAPI.GetBlendShapesAttr();
                pxr::VtArray<pxr::TfToken> BlendShapeNames{};
                BlendShapesAttr.Get(&BlendShapeNames);
                for (const auto& name : BlendShapeNames) {
                    //ED_COUT << " BlendShape Name: " << name.GetString() << std::endl;
                }

                if(BlendShapeQuery.GetNumBlendShapes() != BlendShapeNames.size()){
                    ED_COUT << "ERROR: The length of the BlendShape name is not consistent with the BlendShape";
                }

                for (int32 BlendShapeIndex = 0; BlendShapeIndex < BlendShapeQuery.GetNumBlendShapes(); ++BlendShapeIndex)
                {
                    auto const & UsdBlendShape = BlendShapeQuery.GetBlendShape(BlendShapeIndex);
                    auto BlendShapeName = BlendShapeNames[BlendShapeIndex];
                    std::string PrimaryName = UsdBlendShape.GetPrim().GetName();
                    std::string PrimaryPath = UsdBlendShape.GetPrim().GetPath().GetAsString();
                    ImportData.BlendShapeMap.emplace_back(PrimaryPath, BlendShapeName);

                    ED_COUT << " BlendShape Index " << BlendShapeIndex << " Name " << PrimaryName << " Path " << PrimaryPath << "\n";
                    ConvertBlendShape(
                            UsdBlendShape,
                            StageInfo,
                            NumPointsBeforeThisMesh,
                            InOutUsedMorphTargetNames,
                            OutBlendShapes
                    );
                }

                //if(BlendShapeQuery.GetNumBlendShapes()) {
                //    ED_COUT << "----- Out Start-----\n";
                //    for (auto const &[K, V]: OutBlendShapes) {
                //        ED_COUT << " BlendShape Out " << K << " " << V << "\n";
                //    }
                //    ED_COUT << "----- Out End-----\n";
                //}
            }
        }
    };

    // Actually parse all mesh data
    for (const pxr::SdfPath& SkinnedPrimPath : PathsToSkinnedPrims)
    {
        pxr::UsdGeomMesh SkinnedMesh{Stage->GetPrimAtPath(SkinnedPrimPath)};
        if (!SkinnedMesh)
            continue;

        pxr::UsdPrim ParentPrim = SkinnedMesh.GetPrim().GetParent();
        std::string ParentPrimPath = ParentPrim.GetPath().GetAsString();

        ConvertSkinnedPrim(pxr::UsdGeomMesh{Stage->GetPrimAtPath(SkinnedPrimPath)});
    }

    ED_COUT << " Path To SkeletalMeshImportDataMap " << SkeletalMeshImportDataMap.size() << "\n";
    ED_COUT << " Skeletal Transforms " << SkeletalAnimData << "\n";

    for (auto & Entry : SkeletalMeshImportDataMap){
        ESkeletalMeshImportData& ImportData = Entry.second;
        if (Entry.second.MeshData.Points.size() == 0)
            continue;

        OutPathToSkeletalMeshImportData.emplace(Entry.first, ImportData);
    }

    if(OutBlendShapes.size()){
        for (auto& Pair : OutBlendShapes){
            EUsdBlendShape& BlendShape = Pair.second;
        }
    }

    return true;
}

bool EUsdLuxLightTranslator::Create(EUsdTranslatorContext &context) {
    return true;
}

bool EUsdGeomXformableTranslator::CollapsesChildren(EUsdTranslatorContext &context, ECollapsingType collapsingType){
    ED_COUT << "   - Collapses Children " << context.Prim.GetPath() << "\n";
    bool bCollapsesChildren = false;
    auto PruneFunc = [](const pxr::UsdPrim&) ->bool {return false;};

    pxr::UsdPrim Prim = context.Prim;
    pxr::UsdModelAPI Model{pxr::UsdTyped(Prim)};
    if (Model) {
        EUsdDefaultKind PrimKind = GetDefaultKind(Prim);
        bCollapsesChildren = (Context->KindsToCollapse != EUsdDefaultKind::None) && (EnumHasAnyFlags(Context->KindsToCollapse, PrimKind) || (PrimKind == EUsdDefaultKind::None));
        ED_COUT << "      Is Model " << static_cast<std::underlying_type<EUsdDefaultKind>::type>(PrimKind) << " " << bCollapsesChildren << "\n";

        if (!bCollapsesChildren) {
            bCollapsesChildren = Model.IsKind(pxr::TfToken("prop"), pxr::UsdModelAPI::KindValidationNone);
            ED_COUT << "      Is Prop " << bCollapsesChildren << "\n";
        }

        if(bCollapsesChildren){
            ED_COUT << "    - bCollapses Children\n";
            pxr::VtArray<pxr::UsdPrim> ChildXformPrims = GetAllPrimsOfType(Prim, pxr::TfType::Find<pxr::UsdGeomXformable>(), PruneFunc);
            for(const pxr::UsdPrim& ChildXformPrim : ChildXformPrims){
                //for(auto& schemaTranslator: *Context->RegisteredSchemaTranslators){
                //    if(! schemaTranslator.second.Translator->CanBeCollapsed(context, collapsingType)) {
                //        ED_COUT << " Can Not Be Collapsed: " << ChildXformPrim.GetPath() << "\n";
                //        return false;
                //    }
                //}
                ED_COUT << "     - Child Xform Prim " << ChildXformPrim.GetPath() << "\n";
                auto Translator = Context->GetTranslatorByPrim(ChildXformPrim);
                if(Translator != nullptr){
                    // Usd Child Prim For CanBeCollapsed
                    EUsdTranslatorContext ChildContext{ChildXformPrim, {}};
                    if(! Translator->CanBeCollapsed(ChildContext, collapsingType)){
                        ED_COUT << " Can Not Be Collapsed: " << ChildXformPrim.GetPath() << "\n";
                        return false;
                    }
                }
            }
        }
    }

    if (bCollapsesChildren){
        pxr::VtArray<pxr::UsdPrim> ChildGeomMeshes = GetAllPrimsOfType(Prim, pxr::TfType::Find<pxr::UsdGeomMesh>(), PruneFunc);
        for (const pxr::UsdPrim& ChildGeomMeshPrim : ChildGeomMeshes){
            if (!pxr::UsdGeomMesh{ChildGeomMeshPrim})
                continue;
        }
    }

    ED_COUT << " CollapsesResult " << Prim.GetPath() << " " << bCollapsesChildren << "\n";
    return bCollapsesChildren;
}

bool EUsdGeomXformableTranslator::CanBeCollapsed(EUsdTranslatorContext& context, ECollapsingType collapsingType){
    ED_COUT << " UsdGeomXformable Translator::CanBeCollapsed " << context.Prim.GetPath() << "\n";
    pxr::UsdPrim UsdPrim{context.Prim};
    if (!UsdPrim)
        return false;

    if (IsAnimated(UsdPrim)) {
        ED_COUT << " UsdGeomXformable Translator IsAnimated\n";
        return false;
    }

    if (PrimHasSchema(UsdPrim, EIdentifiers::LiveLinkAPI))
        return false;
    //ED_COUT << "    - Can BeCollapsed " << context.Prim.GetPath() << "\n";
    return true;
}

bool EUsdGeomXformableTranslator::Create(EUsdTranslatorContext &context) {
    auto PrimPath = context.Prim.GetPath();
    ED_COUT << "Xformable Translator Create: " << PrimPath << "\n";

    //if (!CollapsesChildren(context, ECollapsingType::Assets))
    //{
    //    ED_COUT << "  Not Collapses Children " << context.Prim.GetPath() << "\n";
    //    // We only have to create assets when our children are collapsed together
    //    return true;
    //}

    // Step 1
    EUSDImported TempImportContext{};
    EUsdPrimMaterialAssignmentInfo TempMaterialInfo{};
    EUsdMeshConversionOptions Options{};
    Options.MaterialToPrimvarToUVIndex = &Context->MaterialToPrimvarToUVIndex;
    Options.TimeCode = 0.0;

    // Root Transform
    Options.AdditionalTransform = GetPrimTransform(context.Prim, pxr::UsdTimeCode(0.0));

    Options.Converted = &Context->Converted;
    Options.ExtraInfo = &context.ExtraInfo;
    bool bSuccess = ConvertGeomMeshHierarchy(context.Prim, TempImportContext, TempMaterialInfo, Options);

    Context->Imported.PathToMeshImportData.insert(TempImportContext.PathToMeshImportData.begin(), TempImportContext.PathToMeshImportData.end());
    Context->Imported.PathToMeshTransform.insert(TempImportContext.PathToMeshTransform.begin(), TempImportContext.PathToMeshTransform.end());
    Context->Imported.PathToFrameToTransform.insert(TempImportContext.PathToFrameToTransform.begin(), TempImportContext.PathToFrameToTransform.end());

    return bSuccess;
}

bool EUsdGeomPointInstancerTranslator::Create(EUsdTranslatorContext &context) {
    return true;
}

/*
 *  - /meshs
 *  - /meshs/grid2
 *  - /meshs/grid2/mesh_0
 *
 *  We assume when we translate the `mesh_0` we already have the transformation about `/meshs/grid2`
 */
bool EUsdGeomMeshTranslator::Create(EUsdTranslatorContext &context) {
    ED_COUT << "Geom Translator Create: " << context.Prim.GetPath() << "\n";

    if(IsAnimated(context.Prim)){
        ED_COUT << "Is AnimatedMesh\n";
        return ParseAnimatedMesh(context);
    }else{
        ED_COUT << "Is GeomMesh\n";
        return ParseGeomMesh(context);
    }
}

bool EUsdGeomMeshTranslator::ParseGeomMesh(EUsdTranslatorContext &context) {
    // XXX LoadMeshDescriptions
    pxr::UsdTyped UsdMesh = pxr::UsdTyped(context.Prim);
    if (!UsdMesh)
        return false;

    pxr::UsdPrim Prim = UsdMesh.GetPrim();
    pxr::UsdStageRefPtr Stage = Prim.GetStage();
    pxr::SdfPath Path = Prim.GetPrimPath();

    EGeomMeshData TempMeshImportData{};
    EUsdPrimMaterialAssignmentInfo TempMaterialInfo{};
    EUsdMeshConversionOptions Options{};
    Options.MaterialToPrimvarToUVIndex = &Context->MaterialToPrimvarToUVIndex;
    Options.TimeCode = 0.0;

    ConvertGeomMesh(pxr::UsdGeomMesh{UsdMesh}, TempMeshImportData, TempMaterialInfo, Options);

    Context->Imported.PathToMeshImportData[Prim.GetPath().GetAsString()] = TempMeshImportData;

    return false;
}

bool EUsdGeomMeshTranslator::ParseAnimatedMesh(EUsdTranslatorContext &context) {
    //EUsdMeshConversionOptions Options{};
    //Options.MaterialToPrimvarToUVIndex = &Context->MaterialToPrimvarToUVIndex;
    //Options.TimeCode = 0.0;
    //Options.AdditionalTransform = {};

    CreateGeometryCache(context);

    return true;
}

bool EUsdGeomMeshTranslator::CreateGeometryCache(EUsdTranslatorContext &context){
    pxr::UsdTyped UsdMesh = pxr::UsdTyped(context.Prim);

    pxr::UsdPrim Prim = context.Prim.GetPrim();
    pxr::UsdStageRefPtr Stage = Prim.GetStage();
    pxr::SdfPath PrimPath = Prim.GetPrimPath();

    int32 StartFrame = std::floor(Stage->GetStartTimeCode());
    int32 EndFrame = std::ceil(Stage->GetEndTimeCode());

    GetGeometryCacheDataTimeCodeRange(Stage, PrimPath.GetAsString(), StartFrame, EndFrame);
    ED_COUT << "Geometry Cache TimeCode Range " << StartFrame << " " << EndFrame << "\n";
    ED_COUT << "Geometry Cache Prim Path " << PrimPath << "\n";

    pxr::TfToken RenderContextToken = pxr::UsdShadeTokens->universalRenderContext;

    for(int32 FrameStart = StartFrame; FrameStart <= EndFrame; ++FrameStart){
        EGeomMeshData TempMeshImportData{};
        EUsdPrimMaterialAssignmentInfo TempMaterialInfo{};
        EUsdMeshConversionOptions Options{};
        Options.MaterialToPrimvarToUVIndex = &Context->MaterialToPrimvarToUVIndex;
        Options.TimeCode = FrameStart;

        Options.AdditionalTransform = {};
        if(Context->Imported.PathToFrameToTransform.find(PrimPath.GetAsString()) != Context->Imported.PathToFrameToTransform.end()){
            // XXX Apply Transform By Customize
            //Options.AdditionalTransform = Context->Imported.PathToFrameToTransform[PrimPath.GetAsString()][FrameStart];
            Options.AdditionalTransform = {};
            //ED_COUT << "Geometry Cache Found AdditionalTransform\n";
        }

        ConvertGeomMesh(pxr::UsdGeomMesh{UsdMesh}, TempMeshImportData, TempMaterialInfo, Options);

        Context->Imported.PathToFrameToMeshImportData[PrimPath.GetAsString()][FrameStart] = TempMeshImportData;
    }

    return true;
}

bool EUsdGeomMeshTranslator::CollapsesChildren(EUsdTranslatorContext &context, ECollapsingType collapsingType) {
    return false;
}

bool EUsdGeomMeshTranslator::CanBeCollapsed(EUsdTranslatorContext &context, ECollapsingType collapsingType) {
    ED_COUT << " UsdGeomMesh Translator::CanBeCollapsed " << context.Prim.GetPath() << "\n";
    pxr::UsdPrim Prim = context.Prim;
    if(IsGeomMeshALOD(Prim)) {
        ED_COUT << " UsdGeomMesh Translator IsGeomMesh ALOD\n";
        return false;
    }

    return Super::CanBeCollapsed(context, collapsingType);
}

bool EUsdGeomCameraTranslator::Create(EUsdTranslatorContext &context) {
    return true;
}

bool EUsdGroomTranslator::Create(EUsdTranslatorContext &context) {
    ED_COUT << "Groom Translator Create: " << context.Prim.GetPath() << "\n";
    if (!IsGroomPrim(context)) {
        ED_COUT << "Super Create Xformable Translator" << "\n";
        return Super::Create(context);
    }

    // TODO Groom Assets

    return true;
}

bool EUsdGroomTranslator::CollapsesChildren(EUsdTranslatorContext &context, ECollapsingType CollapsingType) {
    if (!IsGroomPrim(context))
        return Super::CollapsesChildren(context, CollapsingType);

    return true;
}

bool EUsdGroomTranslator::CanBeCollapsed(EUsdTranslatorContext &context, ECollapsingType CollapsingType) {
    if (!IsGroomPrim(context))
        return Super::CanBeCollapsed(context, CollapsingType);

    return true;
}

bool EUsdGroomTranslator::IsGroomPrim(EUsdTranslatorContext& context) {
    return PrimHasSchema(context.Prim, EIdentifiers::GroomAPI);
}

bool EBasisCurvesTranslator::Create(EUsdTranslatorContext &context) {
    ED_COUT << "BasisCurves Translator Create\n";
    return true;
}

bool EPointsTranslator::Create(EUsdTranslatorContext &context) {
    ED_COUT << "Points Translator Create\n";
    return true;
}
