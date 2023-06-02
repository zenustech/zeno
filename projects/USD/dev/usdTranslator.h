#ifndef YE_USDTRANSLATOR_H
#define YE_USDTRANSLATOR_H

#include <map>

#include "usdDefinition.h"
#include "usdUtilities.h"

#include "pxr/usd/usdSkel/cache.h"

/*
 * UsdGeomCamera
 * UsdGeomMesh
 * UsdGeomPointInstancer
 * UsdGeomXformable
 *
 * UsdLuxBoundableLightBase
 * UsdLuxNonboundableLightBase
 * UsdSkelRoot
 * UsdShadeMaterial
 *
 */

/*
 *
 */

struct EUsdSchemaTranslator;
struct EPrimTypeTranslator;

struct ESchemaTranslatorContext;
struct EPrimTypeTranslatorContext;

/*
 * Translation Context
 */
struct EUsdTranslationContext{
    EUsdTranslationContext();
    std::map<std::string, std::map<std::string, int32>> MaterialToPrimvarToUVIndex{};
    EUsdDefaultKind KindsToCollapse = EUsdDefaultKind::Component | EUsdDefaultKind::Subcomponent;

    pxr::VtArray<std::pair<std::string, ESchemaTranslatorContext>> RegisteredSchemaTranslators{};
    pxr::VtArray<std::pair<std::string, EPrimTypeTranslatorContext>> RegisteredPrimTypeTranslators{};

    EUSDConverted Converted{};
    EUSDImported Imported{};

    std::shared_ptr<EUsdSchemaTranslator> GetTranslatorByPrim(const pxr::UsdPrim& Prim);
    std::shared_ptr<EUsdSchemaTranslator> GetTranslatorByName(std::string Name);
    std::map<std::string, int32>& FindOrAddPrimvarToUVIndex(std::string primvar);
};

/*
 * - ESchemaTranslatorContext
 *      - EUsdSchemaTranslator - Create(EUsdTranslatorContext)
 *         - EUsdTranslationContext
 */
struct ESchemaTranslatorContext{
    std::shared_ptr<EUsdSchemaTranslator> Translator{};
    pxr::TfToken FixedTypeName{};
};

struct EPrimTypeTranslatorContext{
    std::shared_ptr<EPrimTypeTranslator> Translator{};
    pxr::TfToken FixedTypeName{};
};

/*
 *  Abstract Class
 */
struct EUsdSchemaTranslator{
    EUsdTranslationContext* Context = nullptr;

    virtual bool Create(EUsdTranslatorContext& context);
    virtual bool CollapsesChildren(EUsdTranslatorContext& context, ECollapsingType collapsingType) { return false; }
    virtual bool CanBeCollapsed(EUsdTranslatorContext& context, ECollapsingType collapsingType) { return false; }
};

struct EPrimTypeTranslator{
    EUsdTranslationContext* Context = nullptr;

    virtual bool Create(EUsdTranslatorContext& context);
};

/*
 * Impl Class - PrimType
 */
struct EBasisCurvesTranslator : EPrimTypeTranslator{
    virtual bool Create(EUsdTranslatorContext& context) override;
};

struct EPointsTranslator : EPrimTypeTranslator{
    virtual bool Create(EUsdTranslatorContext& context) override;
};

/*
 * Impl Class - UsdSchema
 */
struct EUsdGeomCameraTranslator : EUsdSchemaTranslator{
    virtual bool Create(EUsdTranslatorContext& context) override;
};

struct EUsdGeomPointInstancerTranslator : EUsdSchemaTranslator{
    virtual bool Create(EUsdTranslatorContext& context) override;
};

struct EUsdGeomXformableTranslator : EUsdSchemaTranslator{
    virtual bool CanBeCollapsed(EUsdTranslatorContext& context, ECollapsingType collapsingType) override;
    virtual bool CollapsesChildren(EUsdTranslatorContext &context, ECollapsingType collapsingType) override;
    virtual bool Create(EUsdTranslatorContext& context) override;
};

struct EUsdGeomMeshTranslator : EUsdGeomXformableTranslator{
    using Super = EUsdGeomXformableTranslator;

    virtual bool Create(EUsdTranslatorContext& context) override;
    virtual bool CollapsesChildren(EUsdTranslatorContext &context, ECollapsingType collapsingType) override;
    virtual bool CanBeCollapsed(EUsdTranslatorContext& context, ECollapsingType collapsingType) override;

    bool ParseGeomMesh(EUsdTranslatorContext &context);
    bool ParseAnimatedMesh(EUsdTranslatorContext &context);
    bool CreateGeometryCache(EUsdTranslatorContext &context);
};

struct EUsdLuxLightTranslator : EUsdSchemaTranslator{
    virtual bool Create(EUsdTranslatorContext& context) override;
};

struct EUsdSkelRootTranslator : EUsdSchemaTranslator{
    pxr::VtArray<EBone> SkeletonBones{};
    ESkeletalAnimData SkeletalAnimData{};
    std::string SkeletonName{};
    std::map<std::string, EUsdBlendShape> NewBlendShapes{};
    std::set<std::string> UsedMorphTargetNames{};
    pxr::UsdSkelCache SkeletonCache{};

    std::map<std::string, ESkeletalMeshImportData> PathToSkeletalMeshImportData;

    virtual bool Create(EUsdTranslatorContext& context) override;
    virtual bool CollapsesChildren(EUsdTranslatorContext &context, ECollapsingType CollapsingType) override { return true; }
    virtual bool CanBeCollapsed(EUsdTranslatorContext &context, ECollapsingType CollapsingType) override { return false; }

    bool LoadAllSkeletalData(EUsdTranslatorContext &context);
    bool CreateSkeletalAnim(EUsdTranslatorContext &context);
};

struct EUsdShadeMaterialTranslator : EUsdSchemaTranslator{
    virtual bool Create(EUsdTranslatorContext& context);
};

struct EUsdGroomTranslator : EUsdGeomXformableTranslator{
    using Super = EUsdGeomXformableTranslator;

    virtual bool Create(EUsdTranslatorContext& context) override;
    virtual bool CollapsesChildren(EUsdTranslatorContext &context, ECollapsingType CollapsingType) override;
    virtual bool CanBeCollapsed(EUsdTranslatorContext &context, ECollapsingType CollapsingType) override;
    bool IsGroomPrim(EUsdTranslatorContext& context);
};

#endif //YE_USDTRANSLATOR_H
