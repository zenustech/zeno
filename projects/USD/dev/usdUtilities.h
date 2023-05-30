#ifndef YE_USDUTILITIES_H
#define YE_USDUTILITIES_H

#include <variant>
#include <string>

#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usdShade/material.h>
#include <pxr/base/vt/types.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/usd/usdSkel/skeletonQuery.h>
#include <pxr/usd/usdSkel/skinningQuery.h>
#include <pxr/usd/usdSkel/blendShape.h>
#include <pxr/usd/usdGeom/mesh.h>

#include "usdDefinition.h"
#include "usdHelper.h"

bool ConvertSkeleton(const pxr::UsdSkelSkeletonQuery& SkeletonQuery,
                     ESkeletalMeshImportData& SkelMeshImportData);

bool ComputeTriangleIndicesSimple(pxr::VtArray <int> &faceVertexCounts,
                                  pxr::VtArray <int> &faceVertexIndices,
                                  const pxr::TfToken &orientation,
                                  pxr::VtVec3iArray& tri_indices);

std::string GetResolvedTexturePath(const pxr::UsdAttribute& TextureAssetPathAttr);

pxr::VtArray<pxr::UsdPrim> GetAllPrimsOfType(const pxr::UsdPrim& StartPrim,
                                             const pxr::TfType& SchemaType,
                                             FunctionPrune PruneChildren,
                                             const pxr::VtArray<pxr::TfType>& ExcludeSchemaTypes = {});

pxr::SdfLayerHandle FindLayerForAttribute(const pxr::UsdAttribute& Attribute,
                                          double TimeCode);

std::string ResolveAssetPath(const pxr::SdfLayerHandle& LayerHandle,
                             const std::string& AssetPath);

void HashShadeMaterial(const pxr::UsdShadeMaterial& UsdShadeMaterial,
                       const pxr::TfToken& RenderContext);

void HashShadeInput(const pxr::UsdShadeInput& ShadeInput);

bool ConvertMaterial(const pxr::UsdShadeMaterial& UsdShadeMaterial,
                     std::map<std::string, int32>& PrimvarToUVIndex);

bool GetVec3ParameterValue(pxr::UsdShadeConnectableAPI& Connectable,
                           const pxr::TfToken& InputName,
                           const pxr::GfVec3f& DefaultValue,
                           FParameterValue& OutValue,
                           std::map<std::string, int32>* PrimvarToUVIndex);

bool GetTextureParameterValue(pxr::UsdShadeInput& ShadeInput,
                              FParameterValue& OutValue,
                              std::map<std::string, int32>* PrimvarToUVIndex);

bool GetPrimvarReaderParameterValue(const pxr::UsdShadeInput& Input,
                                    const pxr::TfToken& PrimvarReaderShaderId,
                                    const std::variant<float, pxr::GfVec2f, pxr::GfVec3f>& DefaultValue,
                                    FParameterValue& OutValue);

bool GetFloatParameterValue(pxr::UsdShadeConnectableAPI& Connectable,
                            const pxr::TfToken& InputName,
                            float DefaultValue,
                            FParameterValue& OutValue,
                            std::map<std::string, int32>* PrimvarToUVIndex = nullptr);

pxr::TfToken GetUsdStageUpAxis(const pxr::UsdStageRefPtr& Stage);

float GetUsdStageMetersPerUnit(const pxr::UsdStageRefPtr& Stage);

ETransform GetWholeXformByPrim(const pxr::UsdPrim& Prim, double TimeCode);

ETransform ConvertAxes(const bool bZUp,
                       const ETransform Transform);

ETransform ConvertMatrix(const EUsdStageInfo& StageInfo,
                         const pxr::GfMatrix4d& InMatrix);

pxr::UsdSkelSkinningQuery CreateSkinningQuery(const pxr::UsdGeomMesh& SkinnedMesh,
                                              const pxr::UsdSkelSkeletonQuery& SkeletonQuery);

bool ConvertSkinnedMesh(const pxr::UsdSkelSkinningQuery& SkinningQuery,
                        const pxr::UsdSkelSkeletonQuery& SkeletonQuery,
                        ESkeletalMeshImportData& SkelMeshImportData,
                        //pxr::VtArray<EUsdPrimMaterialSlot>& MaterialAssignments,
                        std::map<std::string, std::map<std::string, int32>>& MaterialToPrimvarsUVSetNames,
                        const pxr::TfToken& RenderContext,
                        const pxr::TfToken& MaterialPurpose,
                        ESkeletalAnimData& OutAnimData);

bool ConvertBlendShape(const pxr::UsdSkelBlendShape& UsdBlendShape,
                       const EUsdStageInfo& StageInfo,
                       uint32 PointIndexOffset,
                       std::set<std::string>& UsedMorphTargetNames,
                       std::map<std::string, EUsdBlendShape>& OutBlendShapes);

std::string GetPrimvarUsedAsST(pxr::UsdShadeConnectableAPI& UsdUVTexture);

std::string RecursivelySearchForStringValue(pxr::UsdShadeInput Input);

int32 GetPrimvarUVIndex(std::string PrimvarName);

pxr::GfVec3f ConvertVector(const EUsdStageInfo& StageInfo,
                           const pxr::GfVec3f& InValue);

EUsdPrimMaterialAssignmentInfo GetPrimMaterialAssignments(const pxr::UsdPrim& UsdPrim,
                                                          const pxr::UsdTimeCode TimeCode,
                                                          bool bProvideMaterialIndices,
                                                          const pxr::TfToken& RenderContext,
                                                          const pxr::TfToken& MaterialPurpose);

bool IsAnimated(const pxr::UsdPrim& Prim);

bool IsAnimated2(const pxr::UsdPrim& Prim);

bool IsWholeXformAnimated(const pxr::UsdPrim& Prim,
                          int32* OutStartFrame = nullptr,
                          int32* OutEndFrame = nullptr);

bool ConvertGeomMesh(const pxr::UsdGeomMesh& UsdMesh,
                     EGeomMeshData& OutMeshData,
                     EUsdPrimMaterialAssignmentInfo& OutMaterialAssignments,
                     EUsdMeshConversionOptions Options);

EUsdGeomOrientation GetGeometryOrientation(const pxr::UsdGeomMesh& Mesh,
                                           double Time);

std::map<std::string, pxr::UsdGeomPrimvar> GetUVSetPrimvars(const pxr::UsdGeomMesh& UsdMesh);

pxr::VtArray<pxr::UsdGeomPrimvar> GetUVSetPrimvars(const pxr::UsdGeomMesh& UsdMesh,
                                                   std::map<std::string, std::map<std::string, int32>>* MaterialToPrimvarsUVSetNames,
                                                   const EUsdPrimMaterialAssignmentInfo& UsdMeshMaterialAssignmentInfo);

int32 GetPrimValueIndex(const pxr::TfToken& InterpType,
                        const int32 VertexIndex,
                        const int32 VertexInstanceIndex,
                        const int32 PolygonIndex);

void GetGeometryCacheDataTimeCodeRange(pxr::UsdStageRefPtr Stage,
                                       const std::string& PrimPath,
                                       int32& OutStartFrame,
                                       int32& OutEndFrame);

bool ConvertGeomMeshHierarchy(const pxr::UsdPrim& Prim,
                              EUSDImported& OutUSDImported,
                              EUsdPrimMaterialAssignmentInfo& OutMaterialAssignments,
                              EUsdMeshConversionOptions Options);

bool RecursivelyCollapseChildMeshes(const pxr::UsdPrim& Prim,
                                    EUSDImported& OutUSDImported,
                                    EUsdPrimMaterialAssignmentInfo& OutMaterialAssignments,
                                    EUsdMeshConversionOptions& Options, bool bIsFirstPrim);

bool ConvertXformable(const pxr::UsdStageRefPtr& Stage,
                      const pxr::UsdTyped& Schema,
                      ETransform& OutTransform,
                      double EvalTime,
                      bool* bOutResetTransformStack = nullptr);

bool PrimHasSchema(const pxr::UsdPrim& Prim,
                   const pxr::TfToken& SchemaToken);

EUsdDefaultKind GetDefaultKind(const pxr::UsdPrim& Prim);

int32 GetLODIndexFromName(const std::string& Name);

bool DoesPrimContainMeshLODsInternal(const pxr::UsdPrim& Prim);

bool IsGeomMeshALOD(const pxr::UsdPrim& UsdMeshPrim);

bool HasAnimatedVisibility(const pxr::UsdPrim& Prim);

ETransform GetPrimTransform(const pxr::UsdPrim& Prim,
                            pxr::UsdTimeCode TimeCode);

bool CreateUsdBlendShape(const std::string& Name,
                         const pxr::VtArray<pxr::GfVec3f>& PointOffsets,
                         const pxr::VtArray<pxr::GfVec3f>& NormalOffsets,
                         const pxr::VtArray<int>& PointIndices,
                         const EUsdStageInfo& StageInfo,
                         uint32 PointIndexOffset,
                         EUsdBlendShape& OutBlendShape);

bool ConvertSkelAnim(const pxr::UsdSkelSkeletonQuery& InUsdSkeletonQuery,
                     const pxr::VtArray<pxr::UsdSkelSkinningQuery>* InSkinningTargets,
                     //pxr::VtArray<ESkeletalMeshImportData>& InMeshImportData,
                     //std::map<std::string, ESkeletalMeshImportData>& InMeshImportData,
                     pxr::VtArray<EBone>& InSkeletonBones,
                     const EBlendShapeMap* InBlendShapes,
                     ESkeletalAnimData& OutAnimData,
                     const pxr::UsdPrim& RootMotionPrim,
                     float* OutStartOffsetSeconds);

pxr::SdfLayerOffset GetPrimToStageOffset(const pxr::UsdPrim& Prim);

pxr::SdfLayerRefPtr FindLayerForPrim(const pxr::UsdPrim& Prim);

#endif //YE_USDUTILITIES_H
