#include "usdHelper.h"
#include "usdUtilities.h"

#include <iomanip>
#include <iostream>
#include <filesystem>

#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/attribute.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usd/modelAPI.h>
#include <pxr/usd/usd/primCompositionQuery.h>

#include <pxr/usd/usdShade/connectableAPI.h>
#include <pxr/usd/usdShade/materialBindingAPI.h>
#include <pxr/usd/usdShade/types.h>

#include <pxr/usd/usdSkel/bindingAPI.h>
#include <pxr/usd/usdSkel/skinningQuery.h>
#include <pxr/usd/usdSkel/skeletonQuery.h>
#include <pxr/usd/usdSkel/topology.h>
#include <pxr/usd/usdSkel/skeleton.h>
#include <pxr/usd/usdSkel/utils.h>
#include <pxr/usd/usdSkel/cache.h>
#include <pxr/usd/usdSkel/root.h>

#include <pxr/usd/usdLux/lightAPI.h>
#include <pxr/usd/kind/registry.h>
#include <pxr/usd/pcp/mapExpression.h>
#include <pxr/usd/pcp/layerStack.h>

#include <pxr/usd/sdf/spec.h>
#include <pxr/usd/sdf/layerUtils.h>

#include <pxr/usd/usdGeom/tokens.h>
#include <pxr/usd/usdGeom/camera.h>
#include <pxr/usd/usdGeom/xformCache.h>
#include <pxr/usd/usdGeom/metrics.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
#include <pxr/usd/usdGeom/pointInstancer.h>

#include <pxr/usd/ar/resolver.h>
#include <pxr/usd/ar/resolverScopedCache.h>


bool FanTriangulate(pxr::GfVec3i &dst, pxr::VtArray<int> const &src, int offset, int index, int size, bool flip)
{
    if (offset + index + 2 >= size) {
        dst[0] = 0;
        dst[1] = 0;
        dst[2] = 0;
        return false;
    }

    if (flip) {
        dst[0] = src[offset];
        dst[1] = src[offset + index + 2];
        dst[2] = src[offset + index + 1];
    } else {
        dst[0] = src[offset];
        dst[1] = src[offset + index + 1];
        dst[2] = src[offset + index + 2];
    }

    return true;
}

bool ComputeTriangleIndicesSimple(pxr::VtArray<int> &faceVertexCounts,
                                  pxr::VtArray<int> &faceVertexIndices,
                                  const pxr::TfToken& orientation,
                                  pxr::VtVec3iArray &tri_indices){
    int numFaces = faceVertexCounts.size();
    int numVertIndices = faceVertexIndices.size();
    int numTris = 0;

    bool invalidTopology = false;

    for (int i = 0; i < numFaces; ++i) {
        int nv = faceVertexCounts[i] - 2;
        if (nv < 1) {
            invalidTopology = true;
            ED_CERR << " ERROR: InvalidTopology face index: " << i << "\n";
            return invalidTopology;
        } else {
            numTris += nv;
        }
    }

    ED_COUT << "Simple Compute Triangle Indices, Total Num Tris " << numTris << "\n";

    tri_indices.resize(numTris);
    bool flip = (orientation != pxr::UsdGeomTokens->rightHanded);

    // i  -> authored face index [0, numFaces)
    // tv -> triangulated face index [0, numTris)
    // v  -> index to the first vertex (index) for face i
    for (int i = 0, tv = 0, v = 0; i < numFaces; ++i) {
        int nv = faceVertexCounts[i];
        if (nv < 3) {
        } else {
            for (int j = 0; j < nv-2; ++j) {
                //                      dst              src        offset index    size      flip
                if (!FanTriangulate(tri_indices[tv], faceVertexIndices, v, j, numVertIndices, flip)) {
                    invalidTopology = true;
                    return invalidTopology;
                }
                ++tv;
            }
        }
        v += nv;
    }

    return invalidTopology;
}

std::string GetResolvedTexturePath(const pxr::UsdAttribute& TextureAssetPathAttr){
    pxr::SdfAssetPath TextureAssetPath;
    TextureAssetPathAttr.Get<pxr::SdfAssetPath>(&TextureAssetPath);
    ED_COUT << "    TextureAssetPath " << TextureAssetPath << "\n";
    pxr::ArResolver& Resolver = pxr::ArGetResolver();
    std::string AssetIdentifier = TextureAssetPath.GetResolvedPath();
    ED_COUT << "    AssetIdentifier " << AssetIdentifier << "\n";

    // Don't normalize an empty path as the result will be "."
    if (AssetIdentifier.size() > 0)
        AssetIdentifier = Resolver.CreateIdentifier(AssetIdentifier);

    std::string ResolvedTexturePath = AssetIdentifier;

    if (ResolvedTexturePath.empty())
    {
        std::string TexturePath = TextureAssetPath.GetAssetPath();
        if (!TexturePath.empty())
        {
            pxr::SdfLayerRefPtr TextureLayer = FindLayerForAttribute(TextureAssetPathAttr, pxr::UsdTimeCode::EarliestTime().GetValue());
            ResolvedTexturePath = ResolveAssetPath(TextureLayer, TexturePath);
        }
    }

    if (ResolvedTexturePath.empty())
        ED_CERR << "ERROR: Failed to resolve texture path on attribute " << TextureAssetPathAttr.GetPath() << "\n";

    return ResolvedTexturePath;
}

pxr::VtArray<pxr::UsdPrim> GetAllPrimsOfType(
        const pxr::UsdPrim& StartPrim,
        const pxr::TfType& SchemaType,
        FunctionPrune PruneChildren,
        const pxr::VtArray<pxr::TfType>& ExcludeSchemaTypes
)
{
    pxr::VtArray<pxr::UsdPrim> Result;

    // e.g. pxr::UsdTraverseInstanceProxies(UsdPrimIsModel || UsdPrimIsGroup)
    //  Return all model or group children of the specified prim.
    //  If prim is an instance, return the children that pass this predicate as instance proxy prims.
    pxr::UsdPrimRange PrimRange(StartPrim, pxr::UsdTraverseInstanceProxies());
    for (pxr::UsdPrimRange::iterator PrimRangeIt = PrimRange.begin(); PrimRangeIt != PrimRange.end(); ++PrimRangeIt)
    {
        bool bIsExcluded = false;
        for (const pxr::TfType& SchemaToExclude : ExcludeSchemaTypes)
        {
            if (PrimRangeIt->IsA(SchemaToExclude))
            {
                bIsExcluded = true;
                break;
            }
        }
        if (!bIsExcluded && PrimRangeIt->IsA(SchemaType)) {
            Result.emplace_back(*PrimRangeIt);
        }

        if (bIsExcluded || PruneChildren(*PrimRangeIt))
            PrimRangeIt.PruneChildren();
    }
    return Result;
}

pxr::SdfLayerHandle FindLayerForAttribute(const pxr::UsdAttribute& Attribute, double TimeCode)
{
    if (!Attribute)
        return {};

    for (const pxr::SdfPropertySpecHandle& PropertySpec : Attribute.GetPropertyStack(TimeCode))
    {
        if (PropertySpec->HasDefaultValue() || PropertySpec->GetLayer()->GetNumTimeSamplesForPath(PropertySpec->GetPath()) > 0)
            return PropertySpec->GetLayer();
    }

    return {};
}

std::string ResolveAssetPath(const pxr::SdfLayerHandle& LayerHandle, const std::string& AssetPath)
{
    std::string AssetPathToResolve = AssetPath;
    // If it's a UDIM path, we replace the UDIM tag with the start tile
    // TODO UDIM
    //AssetPathToResolve.ReplaceInline( TEXT("<UDIM>"), TEXT("1001") );

    pxr::ArResolverScopedCache ResolverCache;
    pxr::ArResolver& Resolver = pxr::ArGetResolver();

    const std::string RelativePathToResolve =
            LayerHandle
            ? pxr::SdfComputeAssetPathRelativeToLayer(LayerHandle, AssetPathToResolve)
            : AssetPathToResolve;

    std::string AssetIdentifier = Resolver.Resolve(RelativePathToResolve);

    if (AssetIdentifier.size() > 0)
        AssetIdentifier = Resolver.CreateIdentifier(AssetIdentifier);

    std::string ResolvedAssetPath = AssetIdentifier;

    return ResolvedAssetPath;
}

void HashShadeMaterial(const pxr::UsdShadeMaterial& UsdShadeMaterial, const pxr::TfToken& RenderContext){
    pxr::UsdShadeShader SurfaceShader = UsdShadeMaterial.ComputeSurfaceSource({RenderContext});

    if (!SurfaceShader)
        return;

    // TODO HashShadeInput
    for (const pxr::UsdShadeInput &ShadeInput: SurfaceShader.GetInputs()) {
        HashShadeInput(ShadeInput);
    }
}

void HashShadeInput(const pxr::UsdShadeInput& ShadeInput){

}

bool ConvertMaterial(const pxr::UsdShadeMaterial& UsdShadeMaterial, std::map<std::string, int32>& PrimvarToUVIndex){
    pxr::TfToken RenderContextToken = pxr::UsdShadeTokens->universalRenderContext;

    pxr::UsdShadeShader SurfaceShader = UsdShadeMaterial.ComputeSurfaceSource(RenderContextToken);
    if ( !SurfaceShader )
        return false;

    pxr::UsdShadeConnectableAPI Connectable{SurfaceShader};
    FParameterValue ParameterValue;

    // Base color
    {
        if(GetVec3ParameterValue(Connectable, EIdentifiers::DiffuseColor, pxr::GfVec3f(0, 0, 0), ParameterValue, &PrimvarToUVIndex))
        {
            ED_COUT << "\n";
        }
    }
    // Emissive color
    {
        if(GetVec3ParameterValue(Connectable, EIdentifiers::EmissiveColor, pxr::GfVec3f(0, 0, 0), ParameterValue, &PrimvarToUVIndex))
        {
            ED_COUT << "\n";
        }
    }
    // Specular color
    {
        if(GetVec3ParameterValue(Connectable, EIdentifiers::SpecularColor, pxr::GfVec3f(0, 0, 0), ParameterValue, &PrimvarToUVIndex))
        {
            ED_COUT << "\n";
        }
    }
    // Metallic
    {
        if(GetFloatParameterValue(Connectable, EIdentifiers::Metallic, 0.f, ParameterValue, &PrimvarToUVIndex))
        {
            ED_COUT << "\n";
        }
    }
    // Roughness
    {
        if(GetFloatParameterValue(Connectable, EIdentifiers::Roughness, 1.f, ParameterValue, &PrimvarToUVIndex))
        {
            ED_COUT << "\n";
        }
    }
    // Clearcoat
    {
        if(GetFloatParameterValue(Connectable, EIdentifiers::Clearcoat, 0.f, ParameterValue, &PrimvarToUVIndex))
        {
            ED_COUT << "\n";
        }
    }
    // Opacity
    {
        if(GetFloatParameterValue(Connectable, EIdentifiers::Opacity, 1.f, ParameterValue, &PrimvarToUVIndex))
        {
            ED_COUT << "\n";
        }
    }
    // Normal
    {
        if(GetVec3ParameterValue(Connectable, EIdentifiers::Normal, pxr::GfVec3f(0, 0, 1), ParameterValue, &PrimvarToUVIndex))
        {
            ED_COUT << "\n";
        }
    }
    // Displacement
    {
        if(GetFloatParameterValue(Connectable, EIdentifiers::Displacement, 0.f, ParameterValue, &PrimvarToUVIndex))
        {
            ED_COUT << "\n";
        }
    }
    // Refraction
    {
        if(GetFloatParameterValue(Connectable, EIdentifiers::Refraction, 1.5f, ParameterValue, &PrimvarToUVIndex))
        {
            ED_COUT << "\n";
        }
    }
    // Occlusion
    {
        if(GetFloatParameterValue(Connectable, EIdentifiers::Occlusion, 1.0f, ParameterValue, &PrimvarToUVIndex))
        {
            ED_COUT << "\n";
        }
    }

    // TODO Handle world Space normals
    return true;
}

bool GetVec3ParameterValue(pxr::UsdShadeConnectableAPI& Connectable, const pxr::TfToken& InputName, const pxr::GfVec3f& DefaultValue, FParameterValue& OutValue, std::map<std::string, int32>* PrimvarToUVIndex)
{
    pxr::UsdShadeInput Input = Connectable.GetInput( InputName );
    //ED_COUT << "  GetVec3 ParameterValue (" << InputName << ")\n";
    if (!Input) {
        //ED_COUT << "   GetVec3 ParameterValue (" << InputName << ") No Input\n";
        return false;
    }

    // If we have another shader/node connected
    pxr::UsdShadeConnectableAPI Source;
    pxr::TfToken SourceName;
    pxr::UsdShadeAttributeType AttributeType;
    if (pxr::UsdShadeConnectableAPI::GetConnectedSource(Input.GetAttr(), &Source, &SourceName, &AttributeType))
    {
        if (!GetTextureParameterValue(Input, OutValue, PrimvarToUVIndex))
        {
            if (GetPrimvarReaderParameterValue(Input, EIdentifiers::UsdPrimvarReader_float3, DefaultValue, OutValue))
                return true;

            // Check if we have a fallback input that we can use instead, since we don't have a valid texture value
            // TODO Fallback
            if (const pxr::UsdShadeInput FallbackInput = Source.GetInput(EIdentifiers::Fallback)){
                return false;
            }
            // This shader doesn't have anything: Traverse into the input connectable itself
            ED_COUT << "     - Recursion Traverse\n";
            return GetVec3ParameterValue(Source, SourceName, DefaultValue, OutValue, PrimvarToUVIndex);
        }
    }

    // No other node connected, so we must have some value
    else if (InputName != EIdentifiers::Normal)
    {
        pxr::GfVec3f Color = DefaultValue;
        pxr::GfVec3f UsdColor;
        if (Input.Get<pxr::GfVec3f>( &UsdColor))
            Color = UsdColor;

        ED_COUT << "   GetTexture ParameterValue (" << InputName << ") " << Color << "\n";
        OutValue = pxr::GfVec3f(Color);
    }

    return true;
}

bool GetTextureParameterValue(pxr::UsdShadeInput& ShadeInput,  FParameterValue& OutValue, std::map<std::string, int32>* PrimvarToUVIndex){
    const bool bIsNormalInput = ( ShadeInput.GetTypeName() == pxr::SdfValueTypeNames->Normal3f
                                  || ShadeInput.GetTypeName() == pxr::SdfValueTypeNames->Normal3fArray
                                  || ShadeInput.GetBaseName() == EIdentifiers::Normal
    );

    pxr::UsdShadeConnectableAPI UsdUVTextureSource;
    pxr::TfToken UsdUVTextureSourceName;
    pxr::UsdShadeAttributeType UsdUVTextureAttributeType;

    OutValue = 0.0f;

    if (pxr::UsdShadeConnectableAPI::GetConnectedSource(ShadeInput.GetAttr(), &UsdUVTextureSource, &UsdUVTextureSourceName, &UsdUVTextureAttributeType)) {
        ED_COUT << "   Connectable " << UsdUVTextureSource.GetPath() << " Source Name: " << UsdUVTextureSourceName << " Attribute Type: " << UsdUVTextureAttributeType << "\n";
        pxr::UsdShadeInput FileInput;
        if (UsdUVTextureAttributeType == pxr::UsdShadeAttributeType::Output) {
            ED_COUT << "     AttributeType::Output\n";
            FileInput = UsdUVTextureSource.GetInput(EIdentifiers::File);
        } else {
            ED_COUT << "     AttributeType - Else\n";
            FileInput = UsdUVTextureSource.GetInput(UsdUVTextureSourceName);
        }

        // Recursively traverse "inputs:file" connections until we stop finding other connected prims
        pxr::UsdShadeConnectableAPI TextureFileSource;
        pxr::TfToken TextureFileSourceName;
        pxr::UsdShadeAttributeType TextureFileAttributeType;
        while (FileInput) {
            if (pxr::UsdShadeConnectableAPI::GetConnectedSource(FileInput.GetAttr(), &TextureFileSource,
                                                                &TextureFileSourceName, &TextureFileAttributeType)) {
                if (TextureFileAttributeType == pxr::UsdShadeAttributeType::Output)
                    FileInput = TextureFileSource.GetInput(EIdentifiers::File);
                else
                    FileInput = TextureFileSource.GetInput(TextureFileSourceName);
            } else
                break;
        }

        ETextureWrapMode WrapX = ETextureWrapMode::TW_Repeat;
        ETextureWrapMode WrapY = ETextureWrapMode::TW_Repeat;
        if (pxr::UsdAttribute WrapSAttr = UsdUVTextureSource.GetInput(EIdentifiers::WrapS)) {
            pxr::TfToken WrapS;
            if (WrapSAttr.Get(&WrapS)) {
                WrapX = WrapS == EIdentifiers::Repeat
                        ? ETextureWrapMode::TW_Repeat
                        : WrapS == EIdentifiers::Mirror
                          ? ETextureWrapMode::TW_Mirror
                          : ETextureWrapMode::TW_Clamp;
                ED_COUT << "     WrapX: " << WrapX << " " << WrapS << "\n";
            }
        }
        if (pxr::UsdAttribute WrapTAttr = UsdUVTextureSource.GetInput(EIdentifiers::WrapT)) {
            pxr::TfToken WrapT;
            if (WrapTAttr.Get(&WrapT)) {
                WrapY = WrapT == EIdentifiers::Repeat
                        ? ETextureWrapMode::TW_Repeat
                        : WrapT == EIdentifiers::Mirror
                          ? ETextureWrapMode::TW_Mirror
                          : ETextureWrapMode::TW_Clamp;
                ED_COUT << "     WrapY: " << WrapY << " " << WrapT << "\n";
            }
        }

        if (pxr::UsdAttribute ScaleAttr = UsdUVTextureSource.GetInput(EIdentifiers::Scale)) {
            pxr::GfVec4f Scale;
            if (ScaleAttr.Get(&Scale)) {
                ED_COUT << "     Scale: " << Scale << "\n";
            }
        }
        if (pxr::UsdAttribute BiasAttr = UsdUVTextureSource.GetInput(EIdentifiers::Bias)) {
            pxr::GfVec4f Bias;
            if (BiasAttr.Get(&Bias)) {
                ED_COUT << "     Bias: " << Bias << "\n";
            }
        }

        // Check that FileInput is of type Asset
        if (FileInput && FileInput.GetTypeName() == pxr::SdfValueTypeNames->Asset) {
            std::string TexturePath = GetResolvedTexturePath(FileInput.GetAttr());
            ED_COUT << "   TexturePath " << TexturePath << "\n";
            if (!TexturePath.empty())
                OutValue = ETextureParameterValue{TexturePath};
            else
                return false;

            // TODO Make A Texture Data
            bool Texture = true;

            if (Texture) {
                auto PrimvarName = GetPrimvarUsedAsST(UsdUVTextureSource);
                ED_COUT << " PrimvarName " << PrimvarName << "\n";
                int32 UVIndex = 0;
                if (!PrimvarName.empty() && PrimvarToUVIndex) {
                    UVIndex = GetPrimvarUVIndex(PrimvarName);
                    PrimvarToUVIndex->emplace(PrimvarName, UVIndex);
                } else if (PrimvarToUVIndex) {
                    ED_CERR << "ERROR: FailedToParsePrimVar, Failed to find primvar used as st input for texture '" << TexturePath << "' in material '" << "(nullptr)" << "'. Will use UV index 0 instead\n";
                }

                ETextureParameterValue OutTextureValue;
                OutTextureValue.TexturePath = TexturePath;
                OutTextureValue.UVIndex = UVIndex;

                if (UsdUVTextureAttributeType == pxr::UsdShadeAttributeType::Output) {
                    std::string OutputName(UsdUVTextureSourceName.GetString());

                    if (OutputName == "rgb") {
                        OutTextureValue.OutputIndex = 0;
                    } else if (OutputName == "r") {
                        OutTextureValue.OutputIndex = 1;
                    } else if (OutputName == "g") {
                        OutTextureValue.OutputIndex = 2;
                    } else if (OutputName == "b") {
                        OutTextureValue.OutputIndex = 3;
                    } else if (OutputName == "a") {
                        OutTextureValue.OutputIndex = 4;
                    }
                }
                ED_COUT << "  -- Out Texture Value " << OutTextureValue << "\n";
                OutValue = OutTextureValue;
            }
        }
    }

    bool hold_texture_param = std::holds_alternative<ETextureParameterValue>(OutValue);
    return hold_texture_param;
}

bool GetPrimvarReaderParameterValue(const pxr::UsdShadeInput& Input, const pxr::TfToken& PrimvarReaderShaderId, const std::variant<float, pxr::GfVec2f, pxr::GfVec3f>& DefaultValue, FParameterValue& OutValue){
    ED_COUT << "   GetPrimvarReaderParameterValue (" << PrimvarReaderShaderId << ")\n";

    if ( !Input )
        return false;

    const bool bShaderOutputsOnly = true;
    const pxr::UsdShadeAttributeVector ValProdAttrs = Input.GetValueProducingAttributes(bShaderOutputsOnly);
    for (const pxr::UsdAttribute& ValProdAttr : ValProdAttrs)
    {
        const pxr::UsdShadeShader ValProdShader(ValProdAttr.GetPrim());
        if (!ValProdShader)
            continue;

        pxr::TfToken ShaderId;
        if (!ValProdShader.GetShaderId(&ShaderId) || ShaderId != PrimvarReaderShaderId)
            continue;

        const pxr::UsdShadeInput VarnameInput = ValProdShader.GetInput(EIdentifiers::Varname);
        if (!VarnameInput)
            continue;

        // The schema for UsdPrimvarReader specifies that the "varname" input should be
        // string-typed, but some assets might be set up token-typed, so we'll consider
        // either type.
        std::string PrimvarName;
        if (VarnameInput.GetTypeName() == pxr::SdfValueTypeNames->String)
        {
            if (!VarnameInput.Get(&PrimvarName))
                continue;
        }
        else if (VarnameInput.GetTypeName() == pxr::SdfValueTypeNames->Token)
        {
            pxr::TfToken PrimvarNameToken;
            if (!VarnameInput.Get(&PrimvarNameToken))
                continue;

            PrimvarName = PrimvarNameToken.GetString();
        }

        if (PrimvarName.empty())
            continue;

        const pxr::UsdShadeInput FallbackInput = ValProdShader.GetInput(EIdentifiers::Fallback);
        if(PrimvarReaderShaderId == EIdentifiers::UsdPrimvarReader_float3) {

            pxr::GfVec3f FallbackColor = std::get<pxr::GfVec3f>(DefaultValue);
            pxr::GfVec3f UsdFallbackColor;
            if (FallbackInput && FallbackInput.Get<pxr::GfVec3f>(&UsdFallbackColor))
            {
                FallbackColor = UsdFallbackColor;
            }

            ED_COUT << "    FallbackColor " << FallbackColor << " PrimvarName " << PrimvarName << "\n";
            OutValue = EPrimvarReaderParameterValue{PrimvarName, pxr::GfVec3f(FallbackColor)};

            return true;
        }else if(PrimvarReaderShaderId == EIdentifiers::UsdPrimvarReader_float){

            float FallbackFloat = std::get<float>(DefaultValue);
            float UsdFallbackFloat;
            if (FallbackInput && FallbackInput.Get<float>(&UsdFallbackFloat))
            {
                FallbackFloat = UsdFallbackFloat;
            }

            ED_COUT << "    FallbackColor " << FallbackFloat << " PrimvarName " << PrimvarName << "\n";
            OutValue = EPrimvarReaderParameterValue{PrimvarName, float(FallbackFloat)};

            return true;
        }
    }

    return false;
}

bool GetFloatParameterValue(pxr::UsdShadeConnectableAPI& Connectable, const pxr::TfToken& InputName, float DefaultValue, FParameterValue& OutValue, std::map<std::string, int32>* PrimvarToUVIndex)
{
    //ED_COUT << "  GetFloat ParameterValue (" << InputName << ")\n";
    pxr::UsdShadeInput Input = Connectable.GetInput(InputName);
    if (!Input) {
        //ED_COUT << "   GetFloat ParameterValue (" << InputName << ") No Input\n";
        return false;
    }

    // If we have another shader/node connected
    pxr::UsdShadeConnectableAPI Source;
    pxr::TfToken SourceName;
    pxr::UsdShadeAttributeType AttributeType;
    if (pxr::UsdShadeConnectableAPI::GetConnectedSource(Input.GetAttr(), &Source, &SourceName, &AttributeType))
    {
        if (!GetTextureParameterValue(Input, OutValue, PrimvarToUVIndex))
        {
            if (GetPrimvarReaderParameterValue(Input, EIdentifiers::UsdPrimvarReader_float, DefaultValue, OutValue))
                return true;

            // Check if we have a fallback input that we can use instead, since we don't have a valid texture value
            if (const pxr::UsdShadeInput FallbackInput = Source.GetInput(EIdentifiers::Fallback))
            {
                float UsdFallbackFloat;
                if (FallbackInput.Get<float>(&UsdFallbackFloat))
                {
                    ED_COUT << "   Get Fallback Value " << UsdFallbackFloat << "\n";
                    OutValue = UsdFallbackFloat;
                    return true;
                }
            }else{
                ED_COUT << "   - Fallback Input Get Failed\n";
            }

            // Recurse because the attribute may just be pointing at some other attribute that has the data
            // (e.g. when shader input is just "hoisted" and connected to the parent material input)
            return GetFloatParameterValue(Source, SourceName, DefaultValue, OutValue);
        }
    }
    // No other node connected, so we must have some value
    else
    {
        float InputValue = DefaultValue;
        Input.Get<float>(&InputValue);
        ED_COUT << "   GetFloat ParameterValue (" << InputName << ") " << InputValue << "\n";
        OutValue = InputValue;
    }

    return true;
}

bool PrimHasSchema(const pxr::UsdPrim& Prim, const pxr::TfToken& SchemaToken)
{
    if (!Prim)
        return false;

    pxr::TfType Schema = pxr::UsdSchemaRegistry::GetTypeFromSchemaTypeName(SchemaToken);
    return Prim.HasAPI(Schema);
}

pxr::TfToken GetUsdStageUpAxis(const pxr::UsdStageRefPtr& Stage)
{
    return pxr::UsdGeomGetStageUpAxis(Stage);
}

float GetUsdStageMetersPerUnit(const pxr::UsdStageRefPtr& Stage)
{
    auto MetersPerUnit = (float)pxr::UsdGeomGetStageMetersPerUnit(Stage);
    //ED_COUT << "  GetUsdStage Meters PerUnit " << MetersPerUnit << "\n";
    return MetersPerUnit;
}

bool ConvertSkeleton(const pxr::UsdSkelSkeletonQuery& SkeletonQuery,
                     ESkeletalMeshImportData& SkelMeshImportData)
{
    using namespace pxr;

    VtArray<std::string> JointNames;
    VtArray<int32> ParentJointIndices;

    // Retrieve the joint names and parent indices from the skeleton topology
    // GetJointOrder already orders them from parent-to-child
    VtArray<TfToken> JointOrder = SkeletonQuery.GetJointOrder();
    const UsdSkelTopology& SkelTopology = SkeletonQuery.GetTopology();
    for (uint32 Index = 0; Index < SkelTopology.GetNumJoints(); ++Index)
    {
        SdfPath JointPath(JointOrder[Index]);

        std::string JointName = JointPath.GetName();
        JointNames.emplace_back(JointName);

        // Returns the parent joint of the index'th joint, Returns -1 for joints with no parent (roots).
        int ParentIndex = SkelTopology.GetParent(Index);
        ED_COUT << "     - Joint " << JointName << " " << JointPath << " " << ParentIndex << "\n";
        ParentJointIndices.emplace_back(ParentIndex);
    }
    ED_COUT << "    Skeleton: " << JointNames.size() << "\n";
    ED_COUT << "    Skeleton: " << ParentJointIndices.size() << "\n";

    // Skeleton has no joints: Generate a dummy single "Root" bone skeleton
    if (JointNames.size() == 0)
    {
        std::string SkeletonPrimPath = SkeletonQuery.GetPrim().GetPath().GetAsString();
        ED_COUT << "NoBonesInSkeleton, Skeleton prim '" << SkeletonPrimPath << "' has no joints!\n";
        // TODO Add one
        return true;
    }

    // Retrieve the bone transforms to be used as the reference pose
    VtArray<ETransform> BoneTransforms;
    VtArray<GfMatrix4d> JointLocalRestTransforms;
    const bool bAtRest = true;
    bool bJointTransformsComputed = SkeletonQuery.ComputeJointLocalTransforms(&JointLocalRestTransforms, UsdTimeCode::EarliestTime(), bAtRest);
    if (bJointTransformsComputed)
    {
        ED_COUT << " - bJointTransformsComputed " << bJointTransformsComputed << "\n";
        UsdStageWeakPtr Stage = SkeletonQuery.GetSkeleton().GetPrim().GetStage();
        const EUsdStageInfo StageInfo(Stage);

        /*  Right Hand    Y       Coordinate
         *                |
         *                o - X
         *               /
         *              Z
         *
         *   Orient condition 1:   Reference Vector 0,1,0  Up Vector 1,0,0     (Same As Coordinate)
         *
         *  If The Joins-Orientation exactly same with the coordination. The anim3_skel_test.usd will not have
         *   Rotation, Just Translation (0,1,0).
         *
         *   Orient condition 2:   Reference Vector 0,1,0  Up Vector 0,0,1       (Axis Z Rotated 90 degree)
         *                Y
         *                |   X
         *                | /
         *                o - - Z
         *  Just Translation (0,1,0) And Root-Joint Rotated
         *
         *  Orient condition 3:    Reference Vector 0,0,1  Up Vector 0,1,0       (Axis X Rotated -90 degree)
         *                                                                  then (Axis Z Rotated -90 degree)
         *                                           combine as Quat  Rotate -0.5 (240 degree) Axis (0.58,0.58,0.58)
         *                                           || (0.58,0.58,0.58) || approximate 1
         *                Z
         *                |
         *                o - Y
         *               /
         *              X
         *  When The Root-Joint Reference Vector is Z-Axis(0,0,1), after that,
         *   the child-joint will Translation By (0,0,1) if the original child-joint is reference parent-joint as Up Direction.
         *
         *  Orient condition 4:    Reference Vector 0,0,1  Up Vector 1,0,0       (Rotate -90 degree Axis X)
         *
         *                Z
         *                |   Y
         *                | /
         *                o - - X
         *  The Conclusion Same As Condition 3.
         *
         *  Orient condition 5:    Reference Vector 1,0,0  Up Vector 0,1,0       (Axis Z Rotate -90  degree)
         *                                                                  then (Axis X Rotate -180 degree)
         *                X
         *                |   Z
         *                | /
         *                o - - Y
         *    ...
         *  Orient condition 5:    Reference Vector 1,0,0  Up Vector 0,0,1       (Axis X Rotate 90 degree)
         *                                                                  then (Axis Z Rotate 90 degree)
         *                                      combine as Quat  Rotate 0.5 (120 degree) Axis (0.58,0.58,0.58)
         *                X
         *                |
         *                o - Z
         *               /
         *              Y
         *    ...
         */

        for (uint32 Index = 0; Index < JointLocalRestTransforms.size(); ++Index)
        {
            const GfMatrix4d& UsdMatrix = JointLocalRestTransforms[Index];
            ED_COUT << "-- Joint LocalRestTransform UsdMatrix" << UsdMatrix << "\n";
            ED_COUT << "-- Joint Extract " << UsdMatrix.ExtractTranslation() << " " << UsdMatrix.ExtractRotationQuat() << "\n";
            ETransform BoneTransform = ConvertMatrix(StageInfo, UsdMatrix);
            ED_COUT << "-- Joint Bone Transform " << BoneTransform << "\n";
            BoneTransforms.emplace_back(BoneTransform);
        }
    }

    if (JointNames.size() != BoneTransforms.size())
    {
        ED_COUT << "JointNames size are not equal BoneTransforms size\n";
        return false;
    }

    // Store the retrieved data as bones into the SkeletalMeshImportData
    SkelMeshImportData.RefBonesBinary.resize(JointNames.size());
    for (int32 Index = 0; Index < JointNames.size(); ++Index) {
        EBone &Bone = SkelMeshImportData.RefBonesBinary[Index];

        Bone.Name = JointNames[Index];
        Bone.ParentIndex = ParentJointIndices[Index];
        // Increment the number of children each time a bone is referenced as a parent bone; the root has a parent index of -1
        if (Bone.ParentIndex >= 0)
        {
            // The joints are ordered from parent-to-child so the parent will already have been added to the array
            EBone& ParentBone = SkelMeshImportData.RefBonesBinary[Bone.ParentIndex];
            ++ParentBone.NumChildren;
        }

        EJointPos& JointMatrix = Bone.BonePos;
        JointMatrix.Transform = BoneTransforms[Index];
    }
    for (int32 Index = 0; Index < JointNames.size(); ++Index) {
        EBone &Bone = SkelMeshImportData.RefBonesBinary[Index];
        ED_COUT << " Bone " << Bone << "\n";
    }

    return true;
}

ETransform ConvertMatrix(const EUsdStageInfo& StageInfo, const pxr::GfMatrix4d& InMatrix){
    pxr::GfMatrix4f FInMatrix(InMatrix);
    ETransform Transform(FInMatrix);

    if(! ED_RIGHT_COORDINATE) {
        ED_COUT << "    Convert Matrix UpAxis " << StageInfo.UpAxis << "\n";

        Transform = ConvertAxes(StageInfo.UpAxis == EUsdUpAxis::ZAxis, Transform);
    }

    const float EMetersPerUnit = ED_METERS_PER_UNIT;
    if (!IsNearlyEqual(StageInfo.MetersPerUnit, EMetersPerUnit)) {
        //ED_COUT << "    Convert Matrix Unit " << StageInfo.MetersPerUnit << " " << EMetersPerUnit << "\n";
        Transform.ScaleTranslation(StageInfo.MetersPerUnit / EMetersPerUnit);
    }

    return Transform;
}

/*
 *             Y Up                                    Z Up
 *
 *  Right Hand   Y       Houdini          Left Hand    Z         UnrealEngine
 *               |                                     |
 *               o - X                                 o - X
 *              /                                     /
 *             Z                                     Y
 *
 *
 *             Z Up                                    Y Up
 *
 *               Z                                     Y
 *               |  Y                                  |  Z
 *               | /                                   | /
 *               o - - X                               o -  -X
 *
 */
ETransform ConvertAxes(const bool bZUp, const ETransform Transform)
{
    pxr::GfVec3f Translation = {Transform.Translation[0], Transform.Translation[1], Transform.Translation[2]};
    pxr::GfQuatf Rotation = Transform.Rotation;
    pxr::GfVec3f Scale = {Transform.Scale3D[0], Transform.Scale3D[1], Transform.Scale3D[2]};

    ED_COUT << "    - Convert Axes bZUp " << bZUp << "\n";
    ED_COUT << "     * Transform " <<  Transform << "\n";

    if (bZUp)
    {
        //Translation.Y = -Translation.Y;
        //Rotation.X = -Rotation.X;
        //Rotation.Z = -Rotation.Z;

        // In Unreal, Using X,Y,Z as Axis, W as theta
        Translation[1] = -Translation[1];
        const pxr::GfVec3f& rotIma = Rotation.GetImaginary();
        Rotation.SetImaginary(pxr::GfVec3f{-rotIma[0], rotIma[1], -rotIma[2]});
    }
    else
    {
        //Swap( Translation.Y, Translation.Z );
        //Rotation = Rotation.Inverse();
        //Swap( Rotation.Y, Rotation.Z );
        //Swap( Scale.Y, Scale.Z );

        Swap(Translation[1], Translation[2]);
        Rotation = Rotation.GetInverse();
        pxr::GfVec3f rotIma = Rotation.GetImaginary();
        Swap(rotIma[1], rotIma[2]);
        Rotation.SetImaginary(rotIma);
        Swap(Scale[1], Scale[2]);
    }

    ETransform ConvertedTrans = ETransform(Rotation, Translation, Scale);
    ED_COUT << "     * ConvertedTrans " << ConvertedTrans << "\n";
    return ConvertedTrans;
}

// Adapted from UsdSkel_CacheImpl::ReadScope::_FindOrCreateSkinningQuery
pxr::UsdSkelSkinningQuery CreateSkinningQuery(const pxr::UsdGeomMesh& SkinnedMesh, const pxr::UsdSkelSkeletonQuery& SkeletonQuery)
{
    pxr::UsdPrim SkinnedPrim = SkinnedMesh.GetPrim();
    if (!SkinnedPrim)
        return {};

    const pxr::UsdSkelAnimQuery& AnimQuery = SkeletonQuery.GetAnimQuery();

    pxr::UsdSkelBindingAPI SkelBindingAPI{SkinnedPrim};

    return pxr::UsdSkelSkinningQuery(
            SkinnedPrim,
            SkeletonQuery ? SkeletonQuery.GetJointOrder() : pxr::VtTokenArray(),
            AnimQuery ? AnimQuery.GetBlendShapeOrder() : pxr::VtTokenArray(),
            SkelBindingAPI.GetJointIndicesAttr(),
            SkelBindingAPI.GetJointWeightsAttr(),
            SkelBindingAPI.GetSkinningMethodAttr(),
            SkelBindingAPI.GetGeomBindTransformAttr(),
            SkelBindingAPI.GetJointsAttr(),
            SkelBindingAPI.GetBlendShapesAttr(),
            SkelBindingAPI.GetBlendShapeTargetsRel()
    );
}

std::string GetPrimvarUsedAsST(pxr::UsdShadeConnectableAPI& UsdUVTexture){
    pxr::UsdShadeInput StInput = UsdUVTexture.GetInput(EIdentifiers::St);
    if (!StInput) {
        ED_COUT << "Get PrimvarUsedAsST NoStInput\n";
        return {};
    }

    pxr::UsdShadeConnectableAPI PrimvarReader;
    pxr::TfToken PrimvarReaderId;
    pxr::UsdShadeAttributeType AttributeType;

    // Connected to a PrimvarReader
    if (pxr::UsdShadeConnectableAPI::GetConnectedSource(StInput.GetAttr(), &PrimvarReader, &PrimvarReaderId, &AttributeType))
    {
        if (pxr::UsdShadeInput VarnameInput = PrimvarReader.GetInput(EIdentifiers::Varname))
        {
            // PrimvarReader may have a string literal with the primvar name, or it may be connected to
            // e.g. an attribute defined elsewhere
            return RecursivelySearchForStringValue(VarnameInput);
        }
    }

    return {};
}

std::string RecursivelySearchForStringValue(pxr::UsdShadeInput Input)
{
    if (!Input)
        return {};

    if (Input.HasConnectedSource())
    {
        pxr::UsdShadeConnectableAPI Souce;
        pxr::TfToken SourceName;
        pxr::UsdShadeAttributeType SourceType;
        if (pxr::UsdShadeConnectableAPI::GetConnectedSource(Input.GetAttr(), &Souce, &SourceName, &SourceType))
        {
            for (const pxr::UsdShadeInput& DeeperInput : Souce.GetInputs())
            {
                std::string Result = RecursivelySearchForStringValue(DeeperInput);
                if (!Result.empty())
                    return Result;
            }
        }
    }
    else
    {
        std::string StringValue;
        if (Input.Get<std::string>(&StringValue))
            return StringValue;

        pxr::TfToken TokenValue;
        if (Input.Get<pxr::TfToken>(&TokenValue))
            return TokenValue.GetString();
    }

    return {};
}

int32 GetPrimvarUVIndex(std::string PrimvarName)
{
    int32 Index = PrimvarName.size();
    while (Index > 0 && PrimvarName[Index - 1] >= '0' && PrimvarName[Index - 1] <= '9')
        --Index;

    if (Index < PrimvarName.size())
    {
        const int32 Length = PrimvarName.size();
        int32 Count = Index;
        const int32 Skip = Clamp(Count, 0, Length);
        auto UVIndex = std::atoi(PrimvarName.substr(Skip).c_str());
        ED_COUT << " Trim Index " << Index << " UVIndex " << UVIndex << "\n";
        return UVIndex;
    }
    ED_COUT << " UVIndex Default Get " << 0 << "\n";
    return 0;
}

bool ConvertSkinnedMesh(const pxr::UsdSkelSkinningQuery& SkinningQuery,
                        const pxr::UsdSkelSkeletonQuery& SkeletonQuery,
                        ESkeletalMeshImportData& SkelMeshImportData,
                        //pxr::VtArray<EUsdPrimMaterialSlot>& MaterialAssignments,
                        std::map<std::string, std::map<std::string, int32>>& MaterialToPrimvarsUVSetNames,
                        const pxr::TfToken& RenderContext,
                        const pxr::TfToken& MaterialPurpose,
                        ESkeletalAnimData& OutAnimData)
{
    using namespace pxr;

    const UsdPrim& SkinningPrim = SkinningQuery.GetPrim();

    auto WholeTransform = GetWholeXformByPrim(SkinningPrim, UsdTimeCode::EarliestTime().GetValue());
    ED_COUT << "Whole Transform: " << WholeTransform << "\n";
    ED_COUT << "Convert Skinned Mesh: " << SkinningPrim.GetPath() << "\n";
    UsdSkelBindingAPI SkelBindingAPI(SkinningPrim);
    UsdGeomMesh UsdMesh = UsdGeomMesh(SkinningPrim);
    if (!UsdMesh)
        return false;

    SkelMeshImportData.PrimPath = SkinningPrim.GetPath().GetAsString();

    const EUsdStageInfo StageInfo(SkinningPrim.GetStage());
    uint32 NumPoints = 0;
    uint32 NumExistingPoints = SkelMeshImportData.MeshData.Points.size();
    ED_COUT << " Num Existing Points " << NumExistingPoints << "\n";

    VtArray<GfMatrix4d> MeshToSkeletonRestPose;
    {
        bool bSuccess = true;

        // Get world-space restTransforms
        VtArray<GfMatrix4d> WorldSpaceRestTransforms;
        {
            VtArray<GfMatrix4d> JointLocalRestTransforms;
            const bool bAtRest = true;
            bSuccess &= SkeletonQuery.ComputeJointLocalTransforms(&JointLocalRestTransforms, UsdTimeCode::EarliestTime(), bAtRest);
            bSuccess &= UsdSkelConcatJointTransforms(SkeletonQuery.GetTopology(), JointLocalRestTransforms, &WorldSpaceRestTransforms);
            //ED_COUT << "      WorldSpaceRestTransforms " << WorldSpaceRestTransforms << "\n";
            //ED_COUT << "      JointLocalRestTransforms " << JointLocalRestTransforms << "\n";

            OutAnimData.WorldSpaceRestTransforms = WorldSpaceRestTransforms;
            OutAnimData.JointLocalRestTransforms = JointLocalRestTransforms;
        }

        // Get world-space bindTransforms
        VtArray<GfMatrix4d> WorldSpaceBindTransforms;
        bSuccess &= SkeletonQuery.GetJointWorldBindTransforms(&WorldSpaceBindTransforms);
        //ED_COUT << "      WorldSpaceBindTransforms " << WorldSpaceBindTransforms << "\n";

        OutAnimData.WorldSpaceBindTransforms = WorldSpaceBindTransforms;

        if (!bSuccess)
            return false;

        MeshToSkeletonRestPose.resize(WorldSpaceRestTransforms.size());
        for (uint32 Index = 0; Index < WorldSpaceRestTransforms.size(); ++Index)
        {
            MeshToSkeletonRestPose[Index] = WorldSpaceBindTransforms[Index].GetInverse() * WorldSpaceRestTransforms[Index];
        }
        //ED_COUT << "      MeshToSkeletonRestPose " << MeshToSkeletonRestPose << "\n";

        OutAnimData.MeshToSkeletonRestPose = MeshToSkeletonRestPose;
    }

    UsdAttribute PointsAttr = UsdMesh.GetPointsAttr();
    if (PointsAttr)
    {
        VtArray<GfVec3f> UsdPoints;
        PointsAttr.Get(&UsdPoints, UsdTimeCode::Default());
        SkinningQuery.ComputeSkinnedPoints(MeshToSkeletonRestPose, &UsdPoints, UsdTimeCode::EarliestTime());
        NumPoints = UsdPoints.size();
        // FIXEM Point combine
        SkelMeshImportData.MeshData.Points.clear();
        SkelMeshImportData.MeshData.Points.resize(NumPoints);

        for (uint32 PointIndex = 0; PointIndex < NumPoints; ++PointIndex)
        {
            const GfVec3f& Point = UsdPoints[PointIndex];
            // FIXEM need transform?
            //auto Position = WholeTransform.TransformPosition(Point);
            // FIXEM convert point
            //SkelMeshImportData.Points[PointIndex] = ConvertVector(StageInfo, Point);
            SkelMeshImportData.MeshData.Points[PointIndex] = Point;
        }
    }
    ED_COUT << "      NumPoints " << SkelMeshImportData.MeshData.Points.size() << "\n";
    if (NumPoints == 0)
        return false;

    // Convert the face data into SkeletalMeshImportData

    // Face counts
    VtArray<int> FaceCounts;
    UsdAttribute FaceCountsAttribute = UsdMesh.GetFaceVertexCountsAttr();
    if (FaceCountsAttribute)
        FaceCountsAttribute.Get(&FaceCounts, UsdTimeCode::Default());

    // Face indices
    VtArray<int> OriginalFaceIndices;
    UsdAttribute FaceIndicesAttribute = UsdMesh.GetFaceVertexIndicesAttr();
    if (FaceIndicesAttribute)
        FaceIndicesAttribute.Get(&OriginalFaceIndices, UsdTimeCode::Default());

    SkelMeshImportData.MeshData.FaceCounts = FaceCounts;
    SkelMeshImportData.MeshData.FaceIndices = OriginalFaceIndices;

    uint32 NumVertexInstances = static_cast<uint32>(OriginalFaceIndices.size());

    // Normals
    VtArray<GfVec3f> Normals;
    UsdAttribute NormalsAttribute = UsdMesh.GetNormalsAttr();
    if (NormalsAttribute)
    {
        if (NormalsAttribute.Get(&Normals, UsdTimeCode::Default() ) && Normals.size() > 0)
        {
            // Like for points, we need to ensure these normals are in the same coordinate space of the skeleton
            SkinningQuery.ComputeSkinnedNormals(MeshToSkeletonRestPose, &Normals, UsdTimeCode::EarliestTime());
        }
    }

    uint32 NumExistingFaces = SkelMeshImportData.MeshData.Faces.size();
    //uint32 NumExistingWedges = SkelMeshImportData.Wedges.size();

    uint32 NumFaces = FaceCounts.size();
    //SkelMeshImportData.Faces.resize(NumFaces * 2);
    ED_COUT << "      NumFaces " << FaceCounts.size() << "\n";
    ED_COUT << "      NumNormals " << Normals.size() << "\n";
    ED_COUT << "      NumIndices " << OriginalFaceIndices.size() << "\n";

    // Material assignments
    const bool bProvideMaterialIndices = true;
    EUsdPrimMaterialAssignmentInfo LocalInfo = GetPrimMaterialAssignments(
            SkinningPrim,
            pxr::UsdTimeCode::EarliestTime(),
            bProvideMaterialIndices,
            RenderContext,
            MaterialPurpose
    );

    /// Combine identical slots for skeletal meshes
    /// ...

    // Retrieve vertex colors
    UsdGeomPrimvar ColorPrimvar = UsdMesh.GetDisplayColorPrimvar();
    pxr::VtArray<pxr::GfVec3f> Colors;
    EUsdInterpolationMethod DisplayColorInterp = EUsdInterpolationMethod::Constant;
    if (ColorPrimvar)
    {
        pxr::VtArray<pxr::GfVec3f> UsdColors;
        // By flattening the hierarchy of primitives and their associated primvars,
        // it becomes easier to access and manipulate the specific values of a primvar for a given primitive,
        // without having to navigate the hierarchy and account for inheritance.
        if (ColorPrimvar.ComputeFlattened(&UsdColors))
        {
            uint32 NumExpectedColors = 0;
            uint32 NumColors = UsdColors.size();
            pxr::TfToken USDInterpType = ColorPrimvar.GetInterpolation();

            if (USDInterpType == pxr::UsdGeomTokens->uniform)
            {
                NumExpectedColors = NumFaces;
                DisplayColorInterp = EUsdInterpolationMethod::Uniform;
            }
            else if (USDInterpType == pxr::UsdGeomTokens->vertex || USDInterpType == pxr::UsdGeomTokens->varying)
            {
                NumExpectedColors = NumPoints;
                DisplayColorInterp = EUsdInterpolationMethod::Vertex;
            }
            else if (USDInterpType == pxr::UsdGeomTokens->faceVarying)
            {
                NumExpectedColors = NumVertexInstances;
                DisplayColorInterp = EUsdInterpolationMethod::FaceVarying;
            }
            else if (USDInterpType == pxr::UsdGeomTokens->constant)
            {
                NumExpectedColors = 1;
                DisplayColorInterp = EUsdInterpolationMethod::Constant;
            }

            if (NumExpectedColors == NumColors)
            {
                Colors.resize(NumColors);
                for (uint32 Index = 0; Index < NumColors; ++Index)
                {
                    const bool bSRGB = true;
                    Colors.emplace_back(UsdColors[Index]);
                }

                SkelMeshImportData.bHasVertexColors = true;
            }
            else
            {
                ED_COUT << "ERROR: Prim '"<<SkinningPrim.GetPath()<<"' has invalid number of displayColor values for "<< "primvar interpolation type '"<<USDInterpType<<"'! (expected "<<NumExpectedColors<<", found "<<NumColors<<")";
            }
        }
    }
    ED_COUT << "      NumColors " << Colors.size() << "\n";

    // Retrieve vertex opacity
    UsdGeomPrimvar OpacityPrimvar = UsdMesh.GetDisplayOpacityPrimvar();
    pxr::VtArray<float> Opacities;
    EUsdInterpolationMethod DisplayOpacityInterp = EUsdInterpolationMethod::Constant;
    if (OpacityPrimvar)
    {
        pxr::VtArray<float> UsdOpacities;
        if (OpacityPrimvar.ComputeFlattened(&UsdOpacities))
        {
            uint32 NumExpectedOpacities = 0;
            const uint32 NumOpacities = UsdOpacities.size();
            pxr::TfToken USDInterpType = OpacityPrimvar.GetInterpolation();

            if (USDInterpType == pxr::UsdGeomTokens->uniform)
            {
                NumExpectedOpacities = NumFaces;
                DisplayOpacityInterp = EUsdInterpolationMethod::Uniform;
            }
            else if (USDInterpType == pxr::UsdGeomTokens->vertex || USDInterpType == pxr::UsdGeomTokens->varying)
            {
                NumExpectedOpacities = NumPoints;
                DisplayOpacityInterp = EUsdInterpolationMethod::Vertex;
            }
            else if (USDInterpType == pxr::UsdGeomTokens->faceVarying)
            {
                NumExpectedOpacities = NumVertexInstances;
                DisplayOpacityInterp = EUsdInterpolationMethod::FaceVarying;
            }
            else if (USDInterpType == pxr::UsdGeomTokens->constant)
            {
                NumExpectedOpacities = 1;
                DisplayOpacityInterp = EUsdInterpolationMethod::Constant;
            }

            if (NumExpectedOpacities == NumOpacities)
            {
                Opacities.resize(NumOpacities);
                for (uint32 Index = 0; Index < NumOpacities; ++Index)
                {
                    Opacities.emplace_back(UsdOpacities[Index]);
                }

                SkelMeshImportData.bHasVertexColors = true; // We'll need to store these in the vertex colors
            }
            else
            {
                ED_COUT << "ERROR: Prim '"<<SkinningPrim.GetPath()<<"' has invalid number of displayOpacity values for "<< "primvar interpolation type '"<<USDInterpType<<"'! (expected "<<NumExpectedOpacities<<", found "<<NumOpacities<<")";
            }
        }
    }
    ED_COUT << "      NumOpacities " << Opacities.size() << "\n";

    if (Colors.size() < 1)
        Colors.emplace_back(1.0, 1.0, 1.0);
    if (Opacities.size() < 1)
        Opacities.emplace_back(1.0f);

    // UV Set
    SkelMeshImportData.NumTexCoords = 0;
    bool bReverseOrder = GetGeometryOrientation(UsdMesh, UsdTimeCode::Default().GetValue()) == EUsdGeomOrientation::LeftHanded;

    pxr::VtArray<pxr::UsdGeomPrimvar> PrimvarsByUVIndex = GetUVSetPrimvars(UsdMesh, &MaterialToPrimvarsUVSetNames, LocalInfo);
    ED_COUT << "      Prim Material Assignments " << LocalInfo << "\n";
    ED_COUT << "      Primvars ByUVIndex " << PrimvarsByUVIndex << "\n";
    // TODO UV Channel

    //SkelMeshImportData.Wedges.resize((NumExistingFaces + NumFaces) * 6);
    //uint32 NumProcessedFaceVertexIndices = 0;
    //for (uint32 PolygonIndex = NumExistingFaces, LocalIndex = 0; PolygonIndex < NumExistingFaces + NumFaces; ++PolygonIndex, ++LocalIndex) {
    //    const uint32 NumOriginalFaceVertices = FaceCounts[LocalIndex];
    //}

    int32 CurrentPointIndex = 0;
    int32 PointOffset = 0;
    int32 VertexOffset = 0;
    pxr::VtArray<EVertexInstanceID> CornerInstanceIDs{};
    pxr::VtArray<EVertexID> CornerVerticesIDs{};

    for (uint32 PolygonIndex = NumExistingFaces, LocalIndex = 0; PolygonIndex < NumExistingFaces + NumFaces; ++PolygonIndex, ++LocalIndex) {
        const uint32 NumOriginalFaceVertices = FaceCounts[LocalIndex];
        const uint32 NumFinalFaceVertices = NumOriginalFaceVertices;
        CornerInstanceIDs.resize(0);
        CornerVerticesIDs.resize(0);

        for (uint32 CornerIndex = 0; CornerIndex < NumFinalFaceVertices; ++CornerIndex, ++CurrentPointIndex){
            int32 PointIndex = PointOffset + CurrentPointIndex;
            const int32 ControlPointIndex = OriginalFaceIndices[CurrentPointIndex];
            const EVertexID VertexID(VertexOffset + ControlPointIndex);
            const pxr::GfVec3f PointPosition = SkelMeshImportData.MeshData.Points[ControlPointIndex];
            CornerVerticesIDs.emplace_back(VertexID);
            EVertexInstanceID AddedVertexInstanceId = SkelMeshImportData.MeshData.CreateVertexInstance(VertexID);

            if (std::find(CornerVerticesIDs.begin(), CornerVerticesIDs.end(), VertexID) != CornerVerticesIDs.end()) {
                continue;
            }

            CornerInstanceIDs.emplace_back(PointIndex);
        }

        SkelMeshImportData.MeshData.CreatePolygon(CornerInstanceIDs);
    }

    // Convert joint influences into the SkeletalMeshImportData
    VtArray<int> JointIndices;
    VtArray<float> JointWeights;
    SkinningQuery.ComputeVaryingJointInfluences(NumPoints, &JointIndices, &JointWeights);

    // Recompute the joint influences if it's above the limit
    uint32 NumInfluencesPerComponent = SkinningQuery.GetNumInfluencesPerComponent();
    //ED_COUT << "JointIndices " << JointIndices << "\n";
    //ED_COUT << "JointWeights " << JointWeights << "\n";
    ED_COUT << "Num Influences " << NumInfluencesPerComponent << "\n";
    if (NumInfluencesPerComponent > ED_MAX_INFLUENCES_PER_STREAM)
    {
        ED_COUT << "Over Max Num Influences\n";
        UsdSkelResizeInfluences(&JointIndices, NumInfluencesPerComponent, ED_MAX_INFLUENCES_PER_STREAM);
        UsdSkelResizeInfluences(&JointWeights, NumInfluencesPerComponent, ED_MAX_INFLUENCES_PER_STREAM);
        NumInfluencesPerComponent = ED_MAX_INFLUENCES_PER_STREAM;
    }

    /// Combine Influences
    const int32 NumInfluencesBefore = SkelMeshImportData.Influences.size();

    if (JointWeights.size() > (NumPoints - 1) * (NumInfluencesPerComponent - 1))
    {
        uint32 JointIndex = 0;
        SkelMeshImportData.Influences.reserve(NumPoints);
        for (uint32 PointIndex = 0; PointIndex < NumPoints; ++PointIndex)
        {
            //ED_COUT << " Iter Bone Point Index " << PointIndex << "\n";
            // The JointIndices/JointWeights contain the influences data for NumPoints * NumInfluencesPerComponent
            for (uint32 InfluenceIndex = 0; InfluenceIndex < NumInfluencesPerComponent; ++InfluenceIndex, ++JointIndex)
            {
                //ED_COUT << " Iter Bone Influence Index " << InfluenceIndex << " Join Index " << JointIndex << "\n";
                // BoneWeight could be 0 if the actual number of influences were less than NumInfluencesPerComponent for a given point so just ignore it
                float BoneWeight = JointWeights[JointIndex];
                if (BoneWeight != 0.f)
                {
                    ERawBoneInfluence BoneInfluence;
                    BoneInfluence.Weight = BoneWeight;
                    BoneInfluence.BoneIndex = JointIndices[JointIndex];
                    BoneInfluence.VertexIndex = NumExistingPoints + PointIndex;
                    SkelMeshImportData.Influences.emplace_back(BoneInfluence);
                }
            }
        }
    }

    const int32 NumInfluencesAfter = SkelMeshImportData.Influences.size();

    // Joint Mapper
    // In Pixar's Universal Scene Description (USD),
    // the Skeletal Joint Mapper is a tool used to map joints in a skeletal mesh to joints in a rig.
    // When you import a skeletal mesh into USD,
    // it often comes with its own set of joints that define the deformation of the mesh.
    // However, these joints may not be compatible with the joints in the rig that you want to use to animate the mesh.
    // This is where the Skeletal Joint Mapper comes in.

    // If we have a joint mapper this Mesh has an explicit joint ordering,
    // so we need to map joint indices to the skeleton's bone indices
    if (pxr::UsdSkelAnimMapperRefPtr AnimMapper = SkinningQuery.GetJointMapper())
    {
        VtArray<int> SkeletonBoneIndices;
        if (pxr::UsdSkelSkeleton BoundSkeleton = SkelBindingAPI.GetInheritedSkeleton())
        {
            if (pxr::UsdAttribute SkeletonJointsAttr = BoundSkeleton.GetJointsAttr())
            {
                VtArray<TfToken> SkeletonJoints;
                if (SkeletonJointsAttr.Get(&SkeletonJoints))
                {
                    // If the skeleton has N bones, this will just contain { 0, 1, 2, ..., N-1 }
                    int NumSkeletonBones = static_cast<int>(SkeletonJoints.size());
                    for (int SkeletonBoneIndex = 0; SkeletonBoneIndex < NumSkeletonBones; ++SkeletonBoneIndex)
                        SkeletonBoneIndices.push_back(SkeletonBoneIndex);

                    // Use the AnimMapper to produce the indices of the Mesh's joints within the Skeleton's list of joints.
                    // Example: Imagine skeleton had { "Root", "Root/Hip", "Root/Hip/Shoulder", "Root/Hip/Shoulder/Arm", "Root/Hip/Shoulder/Arm/Elbow" }, and so
                    // BoneIndexRemapping was { 0, 1, 2, 3, 4 }. Consider a Mesh that specifies the explicit joints { "Root/Hip/Shoulder", "Root/Hip/Shoulder/Arm" },
                    // and so uses the indices 0 and 1 to refer to Shoulder and Arm. After the Remap call SkeletonBoneIndices will hold { 2, 3 }, as those are the
                    // indices of Shoulder and Arm within the skeleton's bones
                    VtArray<int> BoneIndexRemapping;
                    if (AnimMapper->Remap(SkeletonBoneIndices, &BoneIndexRemapping))
                    {
                        for (int32 AddedInfluenceIndex = NumInfluencesBefore; AddedInfluenceIndex < NumInfluencesAfter; ++AddedInfluenceIndex)
                        {
                            ERawBoneInfluence& Influence = SkelMeshImportData.Influences[AddedInfluenceIndex];
                            Influence.BoneIndex = BoneIndexRemapping[Influence.BoneIndex];
                        }
                    }
                }
            }
        }
    }

    ED_COUT << " Skeletal Mesh ImportData " << SkelMeshImportData << "\n";
    return true;
}

bool ConvertBlendShape(const pxr::UsdSkelBlendShape& UsdBlendShape,
                       const EUsdStageInfo& StageInfo,
                       uint32 PointIndexOffset,
                       std::set<std::string>& UsedMorphTargetNames,
                       std::map<std::string, EUsdBlendShape>& OutBlendShapes)
{
    ED_COUT << "Convert Blend Shape\n";

    // FIXEM read error
    pxr::UsdAttribute OffsetsAttr = UsdBlendShape.GetOffsetsAttr();
    pxr::VtArray<pxr::GfVec3f> Offsets{};
    OffsetsAttr.Get(&Offsets);

    pxr::UsdAttribute IndicesAttr = UsdBlendShape.GetPointIndicesAttr();
    pxr::VtArray<int> PointIndices{};
    IndicesAttr.Get(&PointIndices);

    pxr::UsdAttribute NormalsAttr = UsdBlendShape.GetNormalOffsetsAttr();
    pxr::VtArray<pxr::GfVec3f> Normals{};
    NormalsAttr.Get(&Normals);

    // We need to guarantee blend shapes have unique names
    // TODO Unique name
    std::string PrimaryName = UsdBlendShape.GetPrim().GetName();
    std::string PrimaryPath = UsdBlendShape.GetPrim().GetPath().GetAsString();
    if (OutBlendShapes.find(PrimaryPath) != OutBlendShapes.end())
    {
        ED_COUT << "  BlendShape Already Converted " << PrimaryPath << "\n";
        EUsdBlendShape& ExistingBlendShape = OutBlendShapes[PrimaryPath];
        return true;
    }

    EUsdBlendShape PrimaryBlendShape;
    if (! CreateUsdBlendShape(PrimaryName, Offsets, Normals, PointIndices, StageInfo, PointIndexOffset, PrimaryBlendShape))
        return false;
    UsedMorphTargetNames.emplace(PrimaryBlendShape.Name);

    EBlendShapeMap InbetweenBlendShapes;
    for (const pxr::UsdSkelInbetweenShape& Inbetween : UsdBlendShape.GetInbetweens()){
        if (!Inbetween)
            continue;

        float Weight = 0.0f;
        if (!Inbetween.GetWeight(&Weight))
            continue;

        std::string OrigInbetweenName = Inbetween.GetAttr().GetName();
        std::string InbetweenPath = PrimaryPath + "_" + OrigInbetweenName;
            std::filesystem::path filePath(InbetweenPath);
        //std::string InbetweenName = filePath.stem().string();  // Without extension
        std::string InbetweenName = filePath.filename().string();
        ED_COUT << "  Convert BlendShape OriginalInbetweenName " << OrigInbetweenName << "\n";
        //ED_COUT << "  Convert BlendShape InbetweenPath " << " InbetweenPath " << InbetweenPath << " InbetweenName " << InbetweenName << "\n";

        if (Weight > 1.0f || Weight < 0.0f || IsNearlyZero(Weight) || IsNearlyEqual(Weight, 1.0f))
        {
            //ED_COUT << "  Inbetween shape '"<<InbetweenPath<<"' for blend shape '"<<PrimaryPath<<"' has invalid weight " << Weight << "\n";
            continue;
        }

        pxr::VtArray<pxr::GfVec3f> InbetweenPointsOffsets;
        pxr::VtArray<pxr::GfVec3f> InbetweenNormalOffsets;

        Inbetween.GetOffsets(&InbetweenPointsOffsets);
        Inbetween.GetNormalOffsets(&InbetweenNormalOffsets);

        // Create separate blend shape for the inbetween
        // Now how the inbetween always shares the same point indices as the parent
        EUsdBlendShape InbetweenShape;
        if (! CreateUsdBlendShape(InbetweenName, InbetweenPointsOffsets, InbetweenNormalOffsets, PointIndices, StageInfo, PointIndexOffset, InbetweenShape))
            continue;
        UsedMorphTargetNames.emplace(InbetweenShape.Name);
        InbetweenBlendShapes.emplace(InbetweenPath, InbetweenShape);

        // Keep track of it in the PrimaryBlendShape so we can resolve weights later
        EUsdBlendShapeInbetween ConvertedInbetween;
        ConvertedInbetween.Name = InbetweenShape.Name;
        ConvertedInbetween.Path = InbetweenPath;
        ConvertedInbetween.InbetweenWeight = Weight;
        PrimaryBlendShape.InBetweens.emplace_back(ConvertedInbetween);
    }

    // Sort according to weight so they're easier to resolve later
    std::sort(PrimaryBlendShape.InBetweens.begin(), PrimaryBlendShape.InBetweens.end(),
              [](const EUsdBlendShapeInbetween& Lhs, const EUsdBlendShapeInbetween& Rhs)
              {
                  return Lhs.InbetweenWeight < Rhs.InbetweenWeight;
              });

    PrimaryBlendShape.InBetweenBlendShapes = InbetweenBlendShapes;

    OutBlendShapes.emplace(PrimaryPath, PrimaryBlendShape);

    return true;
}

pxr::GfVec3f ConvertVector(const EUsdStageInfo& StageInfo, const pxr::GfVec3f& InValue){
    pxr::GfVec3f Value = InValue;
    const float MetersPerUnit = ED_METERS_PER_UNIT;
    if (!IsNearlyEqual(StageInfo.MetersPerUnit, MetersPerUnit))
        Value *= StageInfo.MetersPerUnit / MetersPerUnit;

    if(! ED_RIGHT_COORDINATE) {
        const bool bIsZUp = (StageInfo.UpAxis == EUsdUpAxis::ZAxis);
        if (bIsZUp)
            Value[1] = -Value[1];
        else
            Swap(Value[1], Value[2]);
    }

    return Value;
}

EUsdPrimMaterialAssignmentInfo GetPrimMaterialAssignments(const pxr::UsdPrim& UsdPrim,
                                                          const pxr::UsdTimeCode TimeCode,
                                                          bool bProvideMaterialIndices,
                                                          const pxr::TfToken& RenderContext,
                                                          const pxr::TfToken& MaterialPurpose)
{
    if (!UsdPrim)
        return {};

    auto FetchMaterialByComputingBoundMaterial = [&RenderContext, &MaterialPurpose](const pxr::UsdPrim& UsdPrim) -> std::string
    {
        pxr::UsdShadeMaterialBindingAPI BindingAPI(UsdPrim);
        pxr::UsdShadeMaterial ShadeMaterial = BindingAPI.ComputeBoundMaterial(MaterialPurpose);
        if (!ShadeMaterial)
            return {};

        pxr::UsdShadeShader SurfaceShader = ShadeMaterial.ComputeSurfaceSource(RenderContext);
        if (!SurfaceShader)
            return {};

        pxr::UsdPrim ShadeMaterialPrim = ShadeMaterial.GetPrim();
        if (ShadeMaterialPrim)
        {
            pxr::SdfPath Path = ShadeMaterialPrim.GetPath();
            std::string ShadingEngineName = (ShadeMaterialPrim ? ShadeMaterialPrim.GetPrim() : UsdPrim.GetPrim()).GetPrimPath().GetString();
            if(ShadingEngineName.size() > 0)
                return ShadingEngineName;
        }

        return {};
    };

    auto FetchMaterialByMaterialRelationship = [&RenderContext](const pxr::UsdPrim& UsdPrim) -> std::string
    {
        if (pxr::UsdRelationship Relationship = UsdPrim.GetRelationship(pxr::UsdShadeTokens->materialBinding))
        {
            pxr::SdfPathVector Targets;
            Relationship.GetTargets(&Targets);

            if (Targets.size() > 0)
            {
                const pxr::SdfPath& TargetMaterialPrimPath = Targets[0];
                pxr::UsdPrim MaterialPrim = UsdPrim.GetStage()->GetPrimAtPath(TargetMaterialPrimPath);
                pxr::UsdShadeMaterial UsdShadeMaterial{MaterialPrim};
                if (!UsdShadeMaterial)
                {
                    ED_COUT << "IgnoringMaterialInvalid, Ignoring material '" <<TargetMaterialPrimPath<<"' bound to prim '"<<UsdPrim.GetPath() <<"' as it does not possess the UsdShadeMaterial schema\n";
                    return {};
                }

                pxr::UsdShadeShader SurfaceShader = UsdShadeMaterial.ComputeSurfaceSource(RenderContext);
                if (!SurfaceShader)
                {
                    auto RC = RenderContext == pxr::UsdShadeTokens->universalRenderContext ? "universal" : RenderContext.GetString();
                    ED_COUT << "IgnoringMaterialSurface, Ignoring material '" <<TargetMaterialPrimPath<<"' bound to prim '" <<UsdPrim.GetPath()<<"' as it contains no valid surface shader source for render context '"<<RC<<"'\n";
                    return {};
                }

                auto MaterialPrimPath = TargetMaterialPrimPath.GetAsString();
                if (Targets.size() > 1)
                {
                    ED_COUT << "MoreThanOneMaterialBinding, Found more than on material:binding targets on prim '" <<UsdPrim.GetPath()<<"'. The first material ('" <<MaterialPrimPath<<"') will be used, and the rest ignored.\n";
                }

                return MaterialPrimPath;
            }
        }

        return {};
    };

    EUsdPrimMaterialAssignmentInfo Result;

    uint64 NumFaces = 0;
    {
        pxr::UsdGeomMesh Mesh = pxr::UsdGeomMesh(UsdPrim);
        pxr::UsdAttribute FaceCounts = Mesh.GetFaceVertexCountsAttr();
        if (!Mesh || !FaceCounts)
            return Result;

        pxr::VtArray<int> FaceVertexCounts;
        FaceCounts.Get(&FaceVertexCounts, TimeCode);
        NumFaces = FaceVertexCounts.size();
        if (NumFaces < 1)
            return Result;

        if (bProvideMaterialIndices)
            Result.MaterialIndices.resize(NumFaces);
    }

    // Priority 2: material binding directly on the prim
    std::string BoundMaterial = FetchMaterialByComputingBoundMaterial(UsdPrim);
    if(! BoundMaterial.empty())
    {
        EUsdPrimMaterialSlot Slot;
        Slot.MaterialSource = BoundMaterial;
        Slot.AssignmentType = EPrimAssignmentType::MaterialPrim;
        Result.Slots.emplace_back(Slot);

        return Result;
    }

    // Priority 3: material:binding relationship directly on the prim
    std::string TargetMaterial = FetchMaterialByMaterialRelationship(UsdPrim);
    if(! TargetMaterial.empty())
    {
        EUsdPrimMaterialSlot Slot;
        Slot.MaterialSource = TargetMaterial;
        Slot.AssignmentType = EPrimAssignmentType::MaterialPrim;
        Result.Slots.emplace_back(Slot);

        ED_COUT << "UsdPrim Material AssignmentInfo " << Result << "\n";

        return Result;
    }

    // TODO GeomSubset partitions

    // TODO vertex color material using displayColor/displayOpacity information for the entire mesh

    ED_COUT << "Get PrimMaterial Assignments None\n";
    return {};
}

bool IsAnimated(const pxr::UsdPrim& Prim)
{
    bool bHasAttributesTimeSamples = false;
    {
        constexpr bool bIncludeInherited = false;
        pxr::TfTokenVector GeomMeshAttributeNames = pxr::UsdGeomMesh::GetSchemaAttributeNames(bIncludeInherited);
        pxr::TfTokenVector GeomPointBasedAttributeNames = pxr::UsdGeomPointBased::GetSchemaAttributeNames(bIncludeInherited);

        GeomMeshAttributeNames.reserve(GeomMeshAttributeNames.size() + GeomPointBasedAttributeNames.size());
        GeomMeshAttributeNames.insert(GeomMeshAttributeNames.end(), GeomPointBasedAttributeNames.begin(), GeomPointBasedAttributeNames.end());

        for (const pxr::TfToken& AttributeName : GeomMeshAttributeNames)
        {
            const pxr::UsdAttribute& Attribute = Prim.GetAttribute(AttributeName);

            if (Attribute && Attribute.ValueMightBeTimeVarying())
            {
                bHasAttributesTimeSamples = true;
                break;
            }
        }
    }

    return bHasAttributesTimeSamples;
}

bool ConvertGeomMesh(const pxr::UsdGeomMesh& UsdMesh,
                     EGeomMeshData& OutMeshData,
                     EUsdPrimMaterialAssignmentInfo& OutMaterialAssignments,
                     EUsdMeshConversionOptions Options)
{
    if (!UsdMesh)
        return false;

    pxr::UsdPrim UsdPrim = UsdMesh.GetPrim();
    pxr::UsdStageRefPtr Stage = UsdPrim.GetStage();
    const EUsdStageInfo StageInfo(Stage);
    const double TimeCodeValue = Options.TimeCode.GetValue();

    const bool bProvideMaterialIndices = true;
    EUsdPrimMaterialAssignmentInfo LocalInfo = GetPrimMaterialAssignments(
            UsdPrim,
            TimeCodeValue,  // XXX or 0.0
            bProvideMaterialIndices,
            pxr::UsdShadeTokens->universalRenderContext,
            pxr::UsdShadeTokens->allPurpose
    );
    ED_COUT << "  Mat LocalInfo " << LocalInfo << "\n";

    const int32 VertexOffset = OutMeshData.Points.size();
    const int32 VertexInstanceOffset = OutMeshData.VertexInstances.size();
    const int32 PolygonOffset = OutMeshData.Faces.size();

    // Vertex positions
    pxr::VtArray<pxr::GfVec3f>& OutMeshDataVertexPositions = OutMeshData.Points;
    {
        pxr::UsdAttribute Points = UsdMesh.GetPointsAttr();
        if (Points)
        {
            pxr::VtArray< pxr::GfVec3f > PointsArray;
            Points.Get(&PointsArray, TimeCodeValue);

            OutMeshData.ReserveNewVertices(PointsArray.size());
            ED_COUT << " - Transformation " << Options.AdditionalTransform << "\n";
            for (int32 LocalPointIndex = 0; LocalPointIndex < PointsArray.size(); ++LocalPointIndex)
            {
                const pxr::GfVec3f Point = PointsArray[LocalPointIndex];
                // FIXEM
                //auto Position = Options.AdditionalTransform.TransformPosition(ConvertVector(StageInfo, Point));
                auto Position = Options.AdditionalTransform.TransformPosition(Point);
                //ED_COUT << " - " << Point << ", " << Position << " ";
                OutMeshDataVertexPositions[LocalPointIndex] = Position;
            }
            //ED_COUT << "\n";
        }
    }

    uint32 NumSkippedPolygons = 0;
    uint32 NumPolygons = 0;

    // Polygons
    {
        bool bFlipThisGeometry = false;
        if (GetGeometryOrientation(UsdMesh, pxr::UsdTimeCode::Default().GetValue()) == EUsdGeomOrientation::LeftHanded)
            bFlipThisGeometry = !bFlipThisGeometry;

        // Face counts
        pxr::UsdAttribute FaceCountsAttribute = UsdMesh.GetFaceVertexCountsAttr();
        pxr::VtArray<int> FaceCounts;
        if (FaceCountsAttribute)
        {
            FaceCountsAttribute.Get(&FaceCounts, TimeCodeValue);
            NumPolygons = FaceCounts.size();
            OutMeshData.FaceCounts = FaceCounts;
        }
        // Face indices
        pxr::UsdAttribute FaceIndicesAttribute = UsdMesh.GetFaceVertexIndicesAttr();
        pxr::VtArray<int> FaceIndices;
        if (FaceIndicesAttribute) {
            FaceIndicesAttribute.Get(&FaceIndices, TimeCodeValue);
            OutMeshData.FaceIndices = FaceIndices;
        }

        // Normals
        pxr::UsdAttribute NormalsAttribute = UsdMesh.GetNormalsAttr();
        pxr::VtArray<pxr::GfVec3f> Normals;

        if (NormalsAttribute)
            NormalsAttribute.Get(&Normals, TimeCodeValue);
        pxr::TfToken NormalsInterpType = UsdMesh.GetNormalsInterpolation();

        // UVs
        // TODO uv-sets
        pxr::VtArray<EUVSet> UVSets;
        pxr::VtArray<pxr::UsdGeomPrimvar> PrimvarsByUVIndex = GetUVSetPrimvars(UsdMesh, Options.MaterialToPrimvarToUVIndex, LocalInfo);

        if(PrimvarsByUVIndex.empty()){
            auto UVPrimvars = GetUVSetPrimvars(UsdMesh);
            for(auto it = UVPrimvars.begin(); it != UVPrimvars.end(); ++it) {
                PrimvarsByUVIndex.push_back(it->second);
            }
            ED_COUT << " Empty UV Set By Materials, Try Get Directly, Size " << UVPrimvars.size() << "\n";
        }

        int32 HighestAddedUVChannel = 0;
        for (int32 UVChannelIndex = 0; UVChannelIndex < PrimvarsByUVIndex.size(); ++UVChannelIndex){
            pxr::UsdGeomPrimvar& Primvar = PrimvarsByUVIndex[UVChannelIndex];
            ED_COUT << " UV varname " << Primvar.GetBaseName() << "\n";
            if (!Primvar)
            {
                // The user may have name their UV sets 'uv4' and 'uv5', in which case we have no UV sets below 4, so just skip them
                ED_COUT << "Skip UV " << Primvar.GetBaseName() << "\n";
                continue;
            }

            EUVSet UVSet;
            UVSet.InterpType = Primvar.GetInterpolation();
            UVSet.UVSetIndex = UVChannelIndex;

            if (Primvar.IsIndexed())
            {
                if (Primvar.GetIndices(&UVSet.UVIndices, Options.TimeCode) && Primvar.Get(&UVSet.UVs, Options.TimeCode))
                {
                    if (UVSet.UVs.size() > 0)
                    {
                        UVSets.emplace_back(UVSet);
                        HighestAddedUVChannel = UVSet.UVSetIndex;
                        ED_COUT << "UV Set Get ValueIndices " << HighestAddedUVChannel << "\n";
                    }
                }
            }else{
                ED_COUT << " UV Set Primvar NotIndexed\n";
                if (Primvar.Get(&UVSet.UVs))
                {
                    if (UVSet.UVs.size() > 0)
                    {
                        UVSets.emplace_back(UVSet);
                        HighestAddedUVChannel = UVSet.UVSetIndex;
                        ED_COUT << " UV Value Size " << UVSet.UVs.size() << "\n";
                    }
                }
            }
        }

        for(auto const& UVSet: UVSets){
            ED_COUT << " UV " << UVSet << "\n";
        }

        //OutMeshData.ReserveNewPolygons(FaceCounts.size());

        // Vertex color
        pxr::UsdGeomPrimvar ColorPrimvar = UsdMesh.GetDisplayColorPrimvar();
        pxr::TfToken ColorInterpolation = pxr::UsdGeomTokens->constant;
        pxr::VtArray<pxr::GfVec3f> UsdColors;
        if (ColorPrimvar)
        {
            ColorPrimvar.ComputeFlattened(&UsdColors, Options.TimeCode);
            ColorInterpolation = ColorPrimvar.GetInterpolation();
        }

        // Vertex opacity
        pxr::UsdGeomPrimvar OpacityPrimvar = UsdMesh.GetDisplayOpacityPrimvar();
        pxr::TfToken OpacityInterpolation = pxr::UsdGeomTokens->constant;
        pxr::VtArray< float > UsdOpacities;
        if (OpacityPrimvar)
        {
            OpacityPrimvar.ComputeFlattened(&UsdOpacities);
            OpacityInterpolation = OpacityPrimvar.GetInterpolation();
        }

        //std::map<int32, EPolygonGroupID> PolygonGroupMapping;
        pxr::VtArray<EVertexInstanceID> CornerInstanceIDs{};
        pxr::VtArray<EVertexID> CornerVerticesIDs{};
        int32 CurrentVertexInstanceIndex = 0;

        ED_COUT << " Offset " << VertexOffset << " " << VertexInstanceOffset << " " << PolygonOffset << "\n";

        pxr::VtArray<pxr::GfVec3f>& OutMeshDataNormals = OutMeshData.Normals;
        pxr::VtArray<pxr::GfVec4f>& OutMeshDataColors = OutMeshData.Colors;
        std::map<int32, pxr::VtArray<pxr::GfVec2f>>& OutMeshDataUVs = OutMeshData.UVs;

        // Preserve Size
        int32 PSize = 0;
        for (int32 i = 0; i < FaceCounts.size(); ++i) {
            int32 Count = FaceCounts[i];
            for (int32 j = 0; j < Count; ++j, ++PSize) {

            }
        }
        ED_COUT << " Preserve Size " << PSize << "\n";

        OutMeshDataNormals.resize(PSize);
        std::fill(OutMeshDataNormals.begin(), OutMeshDataNormals.end(), pxr::GfVec3f(0.0f, 0.0f, 1.0f));
        OutMeshDataColors.resize(PSize);
        for (const EUVSet& UVSet : UVSets){
            OutMeshDataUVs[UVSet.UVSetIndex].resize(PSize);
        }

        // Handle Polygon
        for (int32 PolygonIndex = 0; PolygonIndex < FaceCounts.size(); ++PolygonIndex){
            int32 PolygonVertexCount = FaceCounts[PolygonIndex];
            CornerInstanceIDs.resize(0);
            CornerVerticesIDs.resize(0);
            //ED_COUT << "  Polygon ID " << PolygonIndex << "\n";
            for (int32 CornerIndex = 0; CornerIndex < PolygonVertexCount; ++CornerIndex, ++CurrentVertexInstanceIndex){
                int32 VertexInstanceIndex = VertexInstanceOffset + CurrentVertexInstanceIndex;
                const EVertexInstanceID VertexInstanceID(VertexInstanceIndex);
                const int32 ControlPointIndex = FaceIndices[CurrentVertexInstanceIndex];
                const EVertexID VertexID(VertexOffset + ControlPointIndex);
                const pxr::GfVec3f VertexPosition = OutMeshDataVertexPositions[VertexID.IDValue];

                //ED_COUT << "   CornerIndex " << CornerIndex << " VertexInstanceIndex " << CurrentVertexInstanceIndex << "\n";
                //ED_COUT << "   VertexInstanceIndex " << VertexInstanceIndex << " ControlPointIndex " << ControlPointIndex << "\n";
                //ED_COUT << "   VertexID " << VertexID.IDValue << "\n";

                if (std::find(CornerVerticesIDs.begin(), CornerVerticesIDs.end(), VertexID) != CornerVerticesIDs.end()) {
                    continue;
                }

                CornerVerticesIDs.emplace_back(VertexID);
                EVertexInstanceID AddedVertexInstanceId = OutMeshData.CreateVertexInstance(VertexID);
                CornerInstanceIDs.emplace_back(VertexInstanceIndex);

                //ED_COUT << "    Added VertexInstanceId " << AddedVertexInstanceId.IDValue << " " << VertexID.IDValue << "\n";

                // Normal
                if (Normals.size() > 0)
                {
                    const int32 NormalIndex = GetPrimValueIndex(NormalsInterpType, ControlPointIndex, CurrentVertexInstanceIndex, PolygonIndex);
                    if (NormalIndex < Normals.size())
                    {
                        const pxr::GfVec3f& Normal = Normals[NormalIndex];
                        // FIXEM
                        //Options.AdditionalTransform.TransformVector(ConvertVector(StageInfo, Normal));
                        Options.AdditionalTransform.TransformVector(Normal);

                        OutMeshDataNormals[AddedVertexInstanceId.IDValue] = Normal;
                    }
                }

                // UV
                for (const EUVSet& UVSet : UVSets)
                {
                    const int32 ValueIndex = GetPrimValueIndex(UVSet.InterpType, ControlPointIndex, CurrentVertexInstanceIndex, PolygonIndex);
                    pxr::GfVec2f UV(0.f, 0.f);
                    if (UVSet.UVIndices.size())
                    {
                        if (UVSet.UVIndices.size() > ValueIndex)
                            UV = UVSet.UVs[UVSet.UVIndices[ValueIndex]];
                    }
                    else if (UVSet.UVs.size() > ValueIndex)
                        UV = UVSet.UVs[ValueIndex];

                    // Flip V for Unreal uv's which match directx
                    pxr::GfVec2f FinalUVVector(UV[0], 1.f - UV[1]);
                    OutMeshDataUVs[UVSet.UVSetIndex][AddedVertexInstanceId.IDValue] = FinalUVVector;
                }

                // Vertex color
                {
                    const int32 ValueIndex = GetPrimValueIndex(ColorInterpolation, ControlPointIndex, CurrentVertexInstanceIndex, PolygonIndex);
                    pxr::GfVec3f UsdColor(1.f, 1.f, 1.f);
                    if (!UsdColors.empty() && UsdColors.size() > ValueIndex)
                        UsdColor = UsdColors[ValueIndex];

                    OutMeshDataColors[AddedVertexInstanceId.IDValue] = std::move(pxr::GfVec4f(UsdColor[0], UsdColor[1], UsdColor[2], 1.f));
                }

                // Vertex opacity
                {
                    const int32 ValueIndex = GetPrimValueIndex(OpacityInterpolation, ControlPointIndex, CurrentVertexInstanceIndex, PolygonIndex);
                    if (!UsdOpacities.empty() && UsdOpacities.size() > ValueIndex)
                    {
                        OutMeshDataColors[AddedVertexInstanceId.IDValue][3] = UsdOpacities[ValueIndex];
                    }
                }
            }

            // Polygon
            OutMeshData.CreatePolygon(CornerInstanceIDs);

            if (CornerVerticesIDs.size() < 3)
            {
                ++NumSkippedPolygons;
                continue;
            }

            if (bFlipThisGeometry)
            {
                for (int32 i = 0; i < CornerInstanceIDs.size() / 2; ++i)
                {
                    Swap(CornerInstanceIDs[i], CornerInstanceIDs[CornerInstanceIDs.size() - i - 1]);
                }
            }
        }
    }

    if (NumPolygons > 0 && NumSkippedPolygons > 0)
    {
        ED_COUT << "Skipped " << NumSkippedPolygons << " out of " << NumPolygons << " faces when parsing the mesh for prim '" << UsdPrim.GetPath() << "', as those faces contained too many repeated vertex indices\n";
    }

    ED_COUT << " Geom Mesh Out " << OutMeshData << "\n";
    ED_COUT << "  - End For " << UsdPrim.GetPath() << "\n";

    return true;
}

EUsdGeomOrientation GetGeometryOrientation(const pxr::UsdGeomMesh& Mesh, double Time)
{
    EUsdGeomOrientation GeomOrientation = EUsdGeomOrientation::RightHanded;

    if (Mesh)
    {
        pxr::UsdAttribute Orientation = Mesh.GetOrientationAttr();
        if(Orientation)
        {
            static pxr::TfToken RightHanded("rightHanded");
            static pxr::TfToken LeftHanded("leftHanded");

            pxr::TfToken OrientationValue;
            Orientation.Get(&OrientationValue, Time);

            GeomOrientation = OrientationValue == LeftHanded ? EUsdGeomOrientation::LeftHanded : EUsdGeomOrientation::RightHanded;
        }
    }

    return GeomOrientation;
}

std::map<std::string, pxr::UsdGeomPrimvar> GetUVSetPrimvars(const pxr::UsdGeomMesh& UsdMesh){
    if (!UsdMesh)
        return {};

    // Collect all primvars that could be used as UV sets
    std::map<std::string, pxr::UsdGeomPrimvar> PrimvarsByName;
    std::map<int32, pxr::VtArray<pxr::UsdGeomPrimvar>> UsablePrimvarsByUVIndex;
    pxr::UsdGeomPrimvarsAPI PrimvarsAPI{UsdMesh};
    for (const pxr::UsdGeomPrimvar& Primvar : PrimvarsAPI.GetPrimvars())
    {
        if (!Primvar || !Primvar.HasValue())
            continue;

        // We only care about primvars that can be used as float2[]. TexCoord2f is included
        const pxr::SdfValueTypeName& TypeName = Primvar.GetTypeName();
        if (!TypeName.GetType().IsA(pxr::SdfValueTypeNames->Float2Array.GetType()))
            continue;

        std::string PrimvarName = Primvar.GetBaseName();
        int32 TargetUVIndex = GetPrimvarUVIndex(PrimvarName);

        ED_COUT << " Get UVSet Primvars " << PrimvarName << " UVIndex " << TargetUVIndex << "\n";

        UsablePrimvarsByUVIndex[TargetUVIndex].emplace_back(Primvar);
        PrimvarsByName[PrimvarName] = Primvar;
    }

    return PrimvarsByName;
}

pxr::VtArray<pxr::UsdGeomPrimvar> GetUVSetPrimvars(const pxr::UsdGeomMesh& UsdMesh, std::map<std::string, std::map<std::string, int32>>* MaterialToPrimvarsUVSetNames, const EUsdPrimMaterialAssignmentInfo& UsdMeshMaterialAssignmentInfo){
    if (!UsdMesh)
        return {};

    std::map<std::string, pxr::UsdGeomPrimvar> PrimvarsByName = GetUVSetPrimvars(UsdMesh);

    pxr::VtArray<pxr::UsdGeomPrimvar> Result{};

    // Collect all primvars that are in fact used by the materials assigned to this mesh
    std::map<int32, pxr::VtArray<pxr::UsdGeomPrimvar>> PrimvarsUsedByAssignedMaterialsPerUVIndex;
    {
        const bool bProvideMaterialIndices = false;
        for (const EUsdPrimMaterialSlot& Slot : UsdMeshMaterialAssignmentInfo.Slots)
        {
            if (Slot.AssignmentType == EPrimAssignmentType::MaterialPrim)
            {
                const std::string& MaterialPath = Slot.MaterialSource;
                if (MaterialToPrimvarsUVSetNames->find(MaterialPath) != MaterialToPrimvarsUVSetNames->end())
                {
                    auto FoundMaterialPrimvars = (*MaterialToPrimvarsUVSetNames)[MaterialPath];
                    for (const std::pair<std::string, int32>& PrimvarAndUVIndex: FoundMaterialPrimvars)
                    {
                        if (PrimvarsByName.find(PrimvarAndUVIndex.first) !=  PrimvarsByName.end())
                        {
                            pxr::UsdGeomPrimvar FoundPrimvar = PrimvarsByName[PrimvarAndUVIndex.first];
                            PrimvarsUsedByAssignedMaterialsPerUVIndex[PrimvarAndUVIndex.second].emplace_back(FoundPrimvar);

                            // Temporary add found Primvar to result.
                            Result.emplace_back(FoundPrimvar);
                        }
                    }
                }
            }
        }
    }

    return Result;
}

int32 GetPrimValueIndex(const pxr::TfToken& InterpType, const int32 VertexIndex, const int32 VertexInstanceIndex, const int32 PolygonIndex)
{
    if (InterpType == pxr::UsdGeomTokens->vertex)
        return VertexIndex;
    else if (InterpType == pxr::UsdGeomTokens->varying)
        return VertexIndex;
    else if (InterpType == pxr::UsdGeomTokens->faceVarying)
        return VertexInstanceIndex;
    else if (InterpType == pxr::UsdGeomTokens->uniform)
        return PolygonIndex;
    else /* if ( InterpType == pxr::UsdGeomTokens->constant ) */
    {
        return 0; // return index 0 for constant or any other unsupported cases
    }
}

void GetGeometryCacheDataTimeCodeRange(pxr::UsdStageRefPtr Stage, const std::string& PrimPath, int32& OutStartFrame, int32& OutEndFrame){
    if (!Stage || PrimPath.empty())
        return;

    pxr::UsdPrim UsdPrim = pxr::UsdPrim{Stage->GetPrimAtPath(pxr::SdfPath{PrimPath})};
    if (!UsdPrim)
        return;

    constexpr bool bIncludeInherited = false;
    pxr::TfTokenVector GeomMeshAttributeNames = pxr::UsdGeomMesh::GetSchemaAttributeNames(bIncludeInherited);
    pxr::TfTokenVector GeomPointBasedAttributeNames = pxr::UsdGeomPointBased::GetSchemaAttributeNames(bIncludeInherited);

    GeomMeshAttributeNames.reserve(GeomMeshAttributeNames.size() + GeomPointBasedAttributeNames.size());
    GeomMeshAttributeNames.insert(GeomMeshAttributeNames.end(), GeomPointBasedAttributeNames.begin(), GeomPointBasedAttributeNames.end());

    double MinStartTimeCode = std::numeric_limits<double>::max();
    double MaxEndTimeCode = std::numeric_limits<double>::min();

    for (const pxr::TfToken& AttributeName : GeomMeshAttributeNames)
    {
        if (const pxr::UsdAttribute& Attribute = UsdPrim.GetAttribute(AttributeName))
        {
            std::vector<double> TimeSamples;
            if (Attribute.GetTimeSamples(&TimeSamples) && TimeSamples.size() > 0)
            {
                MinStartTimeCode = std::min(MinStartTimeCode, TimeSamples[0]);
                MaxEndTimeCode = std::max(MaxEndTimeCode, TimeSamples[TimeSamples.size() - 1]);
            }
        }
    }

    OutStartFrame = (int)std::floor(MinStartTimeCode);
    OutEndFrame = (int)std::ceil(MaxEndTimeCode);

    ED_COUT << "TimeCodeRange " << MinStartTimeCode << " " << MaxEndTimeCode << "\n";
}

bool ConvertGeomMeshHierarchy(const pxr::UsdPrim& Prim, EUSDImported& OutUSDImported, EUsdPrimMaterialAssignmentInfo& OutMaterialAssignments, EUsdMeshConversionOptions Options)
{
    if(!Prim)
        return false;

    EUsdMeshConversionOptions OptionsCopy = Options;
    return RecursivelyCollapseChildMeshes(Prim, OutUSDImported, OutMaterialAssignments, OptionsCopy, true);
}

bool IsWholeXformAnimated(const pxr::UsdPrim& Prim, int32* OutStartFrame, int32* OutEndFrame){
    if (!Prim || !Prim.IsActive())
        return false;

    bool IsAnimated = false;
    double MinStartTimeCode = std::numeric_limits<double>::max();
    double MaxEndTimeCode = std::numeric_limits<double>::min();

    pxr::UsdGeomXformable Xformable(Prim);
    if (Xformable)
    {
        pxr::UsdPrim AncestorPrim = Prim;
        while (AncestorPrim && !AncestorPrim.IsPseudoRoot())
        {
            if (pxr::UsdGeomXformable AncestorXformable{AncestorPrim})
            {
                std::vector<double> AncestorTimeSamples;
                if (AncestorXformable.GetTimeSamples(&AncestorTimeSamples) && AncestorTimeSamples.size() > 0) {
                    IsAnimated = true;
                    if(OutStartFrame != nullptr && OutEndFrame != nullptr){
                        MinStartTimeCode = std::min(MinStartTimeCode, AncestorTimeSamples[0]);
                        MaxEndTimeCode = std::max(MaxEndTimeCode, AncestorTimeSamples[AncestorTimeSamples.size() - 1]);
                    }
                }
            }

            AncestorPrim = AncestorPrim.GetParent();
        }
    }

    if(IsAnimated && OutStartFrame != nullptr && OutEndFrame != nullptr){
        *OutStartFrame = (int)std::floor(MinStartTimeCode);
        *OutEndFrame = (int)std::ceil(MaxEndTimeCode);
    }

    return IsAnimated;
}

ETransform GetWholeXformByPrim(const pxr::UsdPrim& Prim, double TimeCode){
    auto StagePtr = Prim.GetStage();
    auto PrimPath = Prim.GetPath().GetAsString();
    auto SplitPath = SplitStringByDelimiter(PrimPath, "/");

    ED_COUT << " Xform: PrimPath " << PrimPath << " SplitPath Size " << SplitPath.size() << "\n";
    ETransform ChildTransform;
    for(int i=0; i<SplitPath.size(); i++){
        std::string StartPath = "";
        int IterStart = 0;
        while(true){
            if(IterStart > i)
                break;
            StartPath += (SplitPath[IterStart] + "/");

            IterStart++;
        }
        StartPath.erase(StartPath.end()-1);  // Remove the last slash
        if(! StartPath.empty()) {
            ED_COUT << "  --- Path " << StartPath << "\n";
            auto PathPrim = StagePtr->GetPrimAtPath(pxr::SdfPath(StartPath));
            auto Transform = GetPrimTransform(PathPrim, pxr::UsdTimeCode(TimeCode));
            ChildTransform = Transform * ChildTransform;
        }
    }

    return ChildTransform;
}

bool RecursivelyCollapseChildMeshes(const pxr::UsdPrim& Prim, EUSDImported& OutUSDImported, EUsdPrimMaterialAssignmentInfo& OutMaterialAssignments, EUsdMeshConversionOptions& Options, bool bIsFirstPrim)
{
    // Handle Converted
    if (pxr::UsdGeomMesh Mesh = pxr::UsdGeomMesh(Prim)){
        auto StringPath = Mesh.GetPath().GetAsString();
        if(Options.Converted->ConvertedGeomMesh.count(StringPath)){
            ED_COUT << "Converted " << StringPath << "\n";
            return true;
        }
    }

    // Handle Ignored
    if(Options.ExtraInfo != nullptr) {
        auto Ignores = Options.ExtraInfo->ConvertMeshIgnores;
        for (int i = 0; i < Ignores.size(); ++i) {
            auto Path = Prim.GetPath().GetAsString();
            if(Path.find(Ignores[i]) != std::string::npos){
                ED_COUT << "Found Extra Ignore " << Path << "\n";
                return true;
            }
        }
    }

    // TODO Handle Purpose
    ETransform ChildTransform = Options.AdditionalTransform;
    ED_COUT << " --> Recursively Prim " << Prim.GetPath() << "\n";

    if (!bIsFirstPrim){
        ED_COUT<< "  -- IsNot FirstPrim\n";
        if (pxr::UsdGeomImageable UsdGeomImageable = pxr::UsdGeomImageable(Prim))
        {
            if (pxr::UsdAttribute VisibilityAttr = UsdGeomImageable.GetVisibilityAttr())
            {
                pxr::TfToken VisibilityToken;
                if (VisibilityAttr.Get(&VisibilityToken) && VisibilityToken == pxr::UsdGeomTokens->invisible)
                    return true;
            }
        }

        if (pxr::UsdGeomXformable Xformable = pxr::UsdGeomXformable(Prim))
        {
            ETransform LocalChildTransform;
            ConvertXformable(Prim.GetStage(), Xformable, LocalChildTransform, Options.TimeCode.GetValue());
            ED_COUT << " Recursive Transform: Local " << LocalChildTransform << "\n";
            ED_COUT << " Recursive Transform: Rchil " << ChildTransform << "\n";
            ChildTransform = LocalChildTransform * Options.AdditionalTransform;
            ED_COUT << " Recursive Transform: Compu " << ChildTransform << "\n";
        }
    }

    ED_COUT<< "  -- --\n";

    bool bSuccess = true;
    bool bTraverseChildren = true;

    Options.AdditionalTransform = ChildTransform;
    if (pxr::UsdGeomMesh Mesh = pxr::UsdGeomMesh(Prim)){
        ED_COUT << "  Convert Geom Mesh " << Prim.GetPath() << "\n";

        pxr::UsdStageRefPtr StagePtr = Prim.GetStage();

        bool IsPrimHasAnim = IsAnimated2(Prim);
        int32 XformStartFrame = std::floor(StagePtr->GetStartTimeCode());
        int32 XformEndFrame = std::ceil(StagePtr->GetEndTimeCode());
        bool IsPrimHasXformAnim = IsWholeXformAnimated(Prim, &XformStartFrame, &XformEndFrame);

        ED_COUT << "  Animated Prim " << IsPrimHasAnim << " Xform " << IsPrimHasXformAnim << "\n";

        // XXX
        // IsPrimHasAnim        - We will convert it to GeomCache after.
        // IsPrimHasXformAnim   - We will convert the Geom once after.
        //  In here we only get the Transformation about this Geom from range Start to End
        if(IsPrimHasAnim)
        {
            pxr::SdfPath PrimPath = Prim.GetPrimPath();
            int32 StartFrame = std::floor(StagePtr->GetStartTimeCode());
            int32 EndFrame = std::ceil(StagePtr->GetEndTimeCode());
            GetGeometryCacheDataTimeCodeRange(StagePtr, PrimPath.GetAsString(), StartFrame, EndFrame);
            ED_COUT << "   Geom Cache Range " << StartFrame << " " << EndFrame << "\n";
            for(int32 FrameStart = StartFrame; FrameStart <= EndFrame; ++FrameStart){
                OutUSDImported.PathToFrameToTransform[Prim.GetPath().GetAsString()][FrameStart] = GetWholeXformByPrim(Prim, FrameStart);
            }
        }
        else
        {
            if(IsPrimHasXformAnim)
            {
                pxr::SdfPath PrimPath = Prim.GetPrimPath();
                ED_COUT << "   Xform Anim Range " << XformStartFrame << " " << XformEndFrame << "\n";
                for(int32 FrameStart = XformStartFrame; FrameStart <= XformEndFrame; ++FrameStart){
                    OutUSDImported.PathToFrameToTransform[Prim.GetPath().GetAsString()][FrameStart] = GetWholeXformByPrim(Prim, FrameStart);
                }
                Options.AdditionalTransform = {};  // Remove transform and set up by customize
            }

            auto& OutMeshData = OutUSDImported.PathToMeshImportData[Prim.GetPath().GetAsString()];
            bSuccess = ConvertGeomMesh(Mesh, OutMeshData, OutMaterialAssignments, Options);
            OutUSDImported.PathToMeshTransform[Prim.GetPath().GetAsString()] = Options.AdditionalTransform;
        }

        // Mark Converted
        Options.Converted->ConvertedGeomMesh.emplace(Mesh.GetPath().GetAsString());
    }
    else if (pxr::UsdGeomPointInstancer PointInstancer = pxr::UsdGeomPointInstancer{Prim})
    {
        ED_COUT << "  Convert Point Instancer\n";
        // TODO Point Instancer Covert
        bTraverseChildren = false;
    }

    if (bTraverseChildren)
    {
        for (const pxr::UsdPrim& ChildPrim : Prim.GetFilteredChildren(pxr::UsdTraverseInstanceProxies()))
        {
            if (!bSuccess)
                break;

            auto OptionsCopy = Options;

            const bool bChildIsFirstPrim = false;
            ED_COUT << "  Recursive Convert\n";
            bSuccess &= RecursivelyCollapseChildMeshes(
                    ChildPrim,
                    OutUSDImported,
                    OutMaterialAssignments,
                    OptionsCopy,
                    bChildIsFirstPrim
            );
        }
    }

    return bSuccess;
}

bool ConvertXformable(const pxr::UsdStageRefPtr& Stage, const pxr::UsdTyped& Schema, ETransform& OutTransform, double EvalTime, bool* bOutResetTransformStack)
{
    pxr::UsdGeomXformable Xformable(Schema);
    if (!Xformable)
        return false;

    // Transform
    pxr::GfMatrix4d UsdMatrix;
    bool bResetXformStack = false;
    bool* bResetXformStackPtr = bOutResetTransformStack ? bOutResetTransformStack : &bResetXformStack;
    Xformable.GetLocalTransformation(&UsdMatrix, bResetXformStackPtr, EvalTime);
    ED_COUT << "  Convert Xformable - Usd Matrix " << UsdMatrix << "\n";

    EUsdStageInfo StageInfo(Stage);
    OutTransform = ConvertMatrix(StageInfo, UsdMatrix);

    const bool bPrimIsLight = Xformable.GetPrim().HasAPI<pxr::UsdLuxLightAPI>();

    if(! ED_RIGHT_COORDINATE) {
        // TODO Convert The Coordinate
        if (Xformable.GetPrim().IsA<pxr::UsdGeomCamera>() || bPrimIsLight){
            if (StageInfo.UpAxis == EUsdUpAxis::YAxis)
            {
            }
            else
            {
            }
        }
        if (pxr::UsdPrim Parent = Xformable.GetPrim().GetParent()){
            const bool bParentIsLight = Parent.HasAPI<pxr::UsdLuxLightAPI>();
            if (!(*bResetXformStackPtr) && (Parent.IsA< pxr::UsdGeomCamera>() || bParentIsLight))
            {
                if (StageInfo.UpAxis == EUsdUpAxis::YAxis)
                {
                }
                else
                {
                }
            }
        }
    }

    return true;
}

EUsdDefaultKind GetDefaultKind(const pxr::UsdPrim& Prim)
{
    pxr::UsdModelAPI Model{pxr::UsdTyped(Prim)};
    EUsdDefaultKind Result = EUsdDefaultKind::None;

    if (!Model)
        return Result;

    if (Model.IsKind(pxr::KindTokens->model, pxr::UsdModelAPI::KindValidationNone))
        Result |= EUsdDefaultKind::Model;

    if (Model.IsKind( pxr::KindTokens->component, pxr::UsdModelAPI::KindValidationNone))
        Result |= EUsdDefaultKind::Component;

    if (Model.IsKind( pxr::KindTokens->group, pxr::UsdModelAPI::KindValidationNone))
        Result |= EUsdDefaultKind::Group;

    if (Model.IsKind(pxr::KindTokens->assembly, pxr::UsdModelAPI::KindValidationNone))
        Result |= EUsdDefaultKind::Assembly;

    if (Model.IsKind(pxr::KindTokens->subcomponent, pxr::UsdModelAPI::KindValidationNone))
        Result |= EUsdDefaultKind::Subcomponent;

    return Result;
}

int32 GetLODIndexFromName(const std::string& Name)
{
    const std::string LODString = EIdentifiers::LOD.GetString();

    // True if Name does not start with The String
    if (Name.rfind(LODString, 0) != 0)
        return INDEX_NONE;

    // After The String there should be only numbers
    if (Name.find_first_not_of("0123456789", LODString.size()) != std::string::npos)
        return INDEX_NONE;

    const int Base = 10;
    char** EndPtr = nullptr;
    return std::strtol(Name.c_str() + LODString.size(), EndPtr, Base);
}

bool DoesPrimContainMeshLODsInternal(const pxr::UsdPrim& Prim){
    if (!Prim)
        return false;

    const std::string LODString = EIdentifiers::LOD.GetString();

    pxr::UsdVariantSets VariantSets = Prim.GetVariantSets();
    if (!VariantSets.HasVariantSet(LODString))
        return false;

    std::string Selection = VariantSets.GetVariantSet(LODString).GetVariantSelection();
    int32 LODIndex = GetLODIndexFromName(Selection);
    if (LODIndex == INDEX_NONE)
        return false;

    return true;
}

bool IsGeomMeshALOD(const pxr::UsdPrim& UsdMeshPrim)
{
    pxr::UsdGeomMesh UsdMesh{UsdMeshPrim};
    if (!UsdMesh)
        return false;

    return DoesPrimContainMeshLODsInternal(UsdMeshPrim.GetParent());
}

bool HasAnimatedVisibility(const pxr::UsdPrim& Prim)
{
    if (!Prim || !Prim.IsActive())
        return false;

    pxr::UsdGeomImageable Imageable(Prim);
    if (Imageable)
    {
        if (pxr::UsdAttribute Attr = Imageable.GetVisibilityAttr())
        {
            std::vector<double> TimeSamples;
            if (Attr.GetTimeSamples(&TimeSamples) && TimeSamples.size() > 0)
                return true;
        }
    }

    return false;
}

bool IsAnimated2(const pxr::UsdPrim& Prim)
{
    if (!Prim || !Prim.IsActive())
        return false;

    // Xformable
    pxr::UsdGeomXformable Xformable(Prim);
    if (Xformable)
    {
        std::vector<double> TimeSamples;
        Xformable.GetTimeSamples(&TimeSamples);
        if (TimeSamples.size() > 0)
            return true;

        // Xform Stack
        // from ChatGPT
        //  The "xform stack reset" is a feature in USD that allows you to reset the transform hierarchy of an object to its default state.
        if (Xformable.GetResetXformStack())
        {
            pxr::UsdPrim AncestorPrim = Prim.GetParent();
            while (AncestorPrim && !AncestorPrim.IsPseudoRoot())
            {
                if (pxr::UsdGeomXformable AncestorXformable{AncestorPrim})
                {
                    std::vector<double> AncestorTimeSamples;
                    if (AncestorXformable.GetTimeSamples(&AncestorTimeSamples) && AncestorTimeSamples.size() > 0)
                        return true;

                    if (AncestorXformable.GetResetXformStack())
                        break;
                }

                AncestorPrim = AncestorPrim.GetParent();
            }
        }
    }

    // Attributes
    const std::vector<pxr::UsdAttribute>& Attributes = Prim.GetAttributes();
    for (const pxr::UsdAttribute& Attribute : Attributes)
    {
        std::vector<double> TimeSamples;
        if (Attribute.GetTimeSamples(&TimeSamples) && TimeSamples.size() > 0)
            return true;
    }

    // Skeleton
    if (pxr::UsdSkelRoot SkeletonRoot{Prim})
    {
        pxr::UsdSkelCache SkeletonCache;
        SkeletonCache.Populate(SkeletonRoot, pxr::UsdTraverseInstanceProxies());

        std::vector<pxr::UsdSkelBinding> SkeletonBindings;
        SkeletonCache.ComputeSkelBindings(SkeletonRoot, &SkeletonBindings, pxr::UsdTraverseInstanceProxies());

        for (const pxr::UsdSkelBinding& Binding : SkeletonBindings)
        {
            const pxr::UsdSkelSkeleton& Skeleton = Binding.GetSkeleton();
            pxr::UsdSkelSkeletonQuery SkelQuery = SkeletonCache.GetSkelQuery(Skeleton);
            pxr::UsdSkelAnimQuery AnimQuery = SkelQuery.GetAnimQuery();
            if (!AnimQuery)
                continue;

            std::vector<double> JointTimeSamples;
            std::vector<double> BlendShapeTimeSamples;
            if ((AnimQuery.GetJointTransformTimeSamples(&JointTimeSamples) && JointTimeSamples.size() > 0) ||
                 (AnimQuery.GetBlendShapeWeightTimeSamples(&BlendShapeTimeSamples) && BlendShapeTimeSamples.size() > 0))
            {
                return true;
            }

            // We only parse Skeletons and SkelAnimations from the first skeletal binding of a SkelRoot, so
            // if that one is not animated then this entire SkelRoot is not animated to us either (for now)
            break;
        }
    }

    return false;
}

ETransform GetPrimTransform(const pxr::UsdPrim& Prim, pxr::UsdTimeCode TimeCode){
    if(pxr::UsdGeomXformable Xformable = pxr::UsdGeomXformable(Prim)){
        ETransform TempOutTransform;
        ConvertXformable(Prim.GetStage(), Xformable, TempOutTransform, TimeCode.GetValue());
        ED_COUT << "   Xformable Transform: " << TempOutTransform << "\n";
        return TempOutTransform;
    }else{
        return {};
    }
}

bool CreateUsdBlendShape(const std::string& Name,
                         const pxr::VtArray<pxr::GfVec3f>& PointOffsets,
                         const pxr::VtArray<pxr::GfVec3f>& NormalOffsets,
                         const pxr::VtArray<int>& PointIndices,
                         const EUsdStageInfo& StageInfo,
                         uint32 PointIndexOffset,
                         EUsdBlendShape& OutBlendShape)
{
    uint32 NumOffsets = PointOffsets.size();
    uint32 NumIndices = PointIndices.size();
    uint32 NumNormals = NormalOffsets.size();

    ED_COUT << " Create Usd BlendShape Name " << Name << " \n";
    ED_COUT << " Create Usd BlendShape PointOffsets Size " << NumOffsets << " \n";
    ED_COUT << " Create Usd BlendShape PointIndices Size " << NumIndices << " \n";
    ED_COUT << " Create Usd BlendShape NormalOffsets Size " << NumNormals << " \n";
    //ED_COUT << " Create Usd BlendShape PointIndex Offset " << PointIndexOffset << " \n";

    // TODO Mismatching Handle

    if (NumNormals > 0)
        OutBlendShape.bHasAuthoredTangents = true;

    OutBlendShape.Name = Name;

    // Prepare the indices of the corresponding base points/normals for every local point/normal we have
    pxr::VtArray<int32> BaseIndices;
    BaseIndices.reserve(NumOffsets);
    if (NumIndices == 0)
    {
        // If we have no indices it means we have information for all of our local points/normals
        for (uint32 BaseIndex = PointIndexOffset; BaseIndex < PointIndexOffset + NumOffsets; ++BaseIndex )
            BaseIndices.emplace_back(static_cast<int32>(BaseIndex));
    }
    else
    {
        // If we have indices it means our morph target only affects a subset of the base vertices
        for (uint32 LocalIndex = 0; LocalIndex < NumOffsets; ++LocalIndex)
        {
            int32 BaseIndex = PointIndices[LocalIndex] + static_cast<int32>(PointIndexOffset);
            BaseIndices.emplace_back(BaseIndex);
        }
    }

    OutBlendShape.Vertices.resize(NumOffsets);
    for (uint32 OffsetIndex = 0; OffsetIndex < NumOffsets; ++OffsetIndex)
    {
        // FIXEM convert blend shape point offset
        //auto Offset = ConvertVector(StageInfo, PointOffsets[OffsetIndex]);
        auto Offset = PointOffsets[OffsetIndex];
        //auto Normal = OutBlendShape.bHasAuthoredTangents
        //                         ? ConvertVector(StageInfo, NormalOffsets[OffsetIndex])
        //                         : pxr::GfVec3f(0, 0, 0);
        auto Normal = OutBlendShape.bHasAuthoredTangents
                                 ? NormalOffsets[OffsetIndex]
                                 : pxr::GfVec3f(0, 0, 0);

        EMorphTargetDelta& ModifiedVertex = OutBlendShape.Vertices[OffsetIndex];
        ModifiedVertex.PositionDelta = Offset;
        ModifiedVertex.TangentZDelta = Normal;
        ModifiedVertex.SourceIdx = BaseIndices[OffsetIndex];
    }

    //ED_COUT << " Out BlendShape Data " << OutBlendShape << "\n";

    return true;
}

bool ConvertSkelAnim(const pxr::UsdSkelSkeletonQuery& InUsdSkeletonQuery,
                     const pxr::VtArray<pxr::UsdSkelSkinningQuery>* InSkinningTargets,
                     //pxr::VtArray<ESkeletalMeshImportData>& InMeshImportData,
                     //std::map<std::string, ESkeletalMeshImportData>& InMeshImportData,
                     pxr::VtArray<EBone>& InSkeletonBones,
                     const EBlendShapeMap* InBlendShapes,
                     ESkeletalAnimData& OutAnimData,
                     const pxr::UsdPrim& RootMotionPrim,
                     float* OutStartOffsetSeconds)
{
    ED_COUT << " Convert Skel Anim " << RootMotionPrim.GetPath() << "\n";
    if (!InUsdSkeletonQuery)
        return false;

    pxr::UsdSkelAnimQuery AnimQuery = InUsdSkeletonQuery.GetAnimQuery();
    if (!AnimQuery)
        return false;

    pxr::UsdPrim SkelAnimPrim = AnimQuery.GetPrim();
    pxr::SdfLayerOffset Offset = GetPrimToStageOffset(pxr::UsdPrim{SkelAnimPrim});

    pxr::SdfLayerRefPtr SkelAnimPrimLayer = FindLayerForPrim(SkelAnimPrim);
    double LayerTimeCodesPerSecond = SkelAnimPrimLayer->GetTimeCodesPerSecond();

    pxr::UsdStageWeakPtr Stage(InUsdSkeletonQuery.GetPrim().GetStage());
    EUsdStageInfo StageInfo{Stage};
    double StageTimeCodesPerSecond = Stage->GetTimeCodesPerSecond();
    if (IsNearlyZero(StageTimeCodesPerSecond))
    {
        ED_COUT << "TimeCodesPerSecondIsZero, Cannot bake skeletal animations as the stage has timeCodesPerSecond set to zero!\n";
        return false;
    }

    InUsdSkeletonQuery.GetJointOrder().size();

    std::vector<double> UsdJointTransformTimeSamples;
    AnimQuery.GetJointTransformTimeSamples(&UsdJointTransformTimeSamples);
    int32 NumJointTransformSamples = UsdJointTransformTimeSamples.size();
    ED_COUT << " Stage TimeCodes PerSecond " << StageTimeCodesPerSecond << " Num JointTransform Samples " << NumJointTransformSamples << "\n";
    double FirstJointSampleTimeCode;
    double LastJointSampleTimeCode;
    if (UsdJointTransformTimeSamples.size() > 0)
    {
        const std::vector<double>& JointTransformTimeSamples = UsdJointTransformTimeSamples;
        FirstJointSampleTimeCode = JointTransformTimeSamples[0];
        LastJointSampleTimeCode = JointTransformTimeSamples[JointTransformTimeSamples.size() - 1];
    }
    ED_COUT << " JointSampleTimeCode " << FirstJointSampleTimeCode << " " << LastJointSampleTimeCode << "\n";

    std::vector<double> UsdBlendShapeTimeSamples;
    AnimQuery.GetBlendShapeWeightTimeSamples(&UsdBlendShapeTimeSamples);
    int32 NumBlendShapeSamples = UsdBlendShapeTimeSamples.size();
    double FirstBlendShapeSampleTimeCode;
    double LastBlendShapeSampleTimeCode;
    if (UsdBlendShapeTimeSamples.size() > 0)
    {
        const std::vector<double>& BlendShapeTimeSamples = UsdBlendShapeTimeSamples;
        FirstBlendShapeSampleTimeCode = BlendShapeTimeSamples[0];
        LastBlendShapeSampleTimeCode = BlendShapeTimeSamples[BlendShapeTimeSamples.size() - 1];
    }
    ED_COUT << " BlendShapeSampleTimeCode " << FirstBlendShapeSampleTimeCode << " " << LastBlendShapeSampleTimeCode << "\n";

    std::vector<double> UsdRootMotionPrimTimeSamples;
    double FirstRootMotionTimeCode;
    double LastRootMotionTimeCode;
    pxr::UsdGeomXformable RootMotionXformable;
    {
        // Note how we don't care whether the root motion is animated or not and will use RootMotionXformable
        // regardless, to have a similar effect in case its just a single non-animated transform
        RootMotionXformable = pxr::UsdGeomXformable{RootMotionPrim};
        if (RootMotionXformable)
        {
            std::vector<double> UsdTimeSamples;
            if (RootMotionXformable.GetTimeSamples(&UsdTimeSamples))
            {
                if (UsdTimeSamples.size() > 0)
                {
                    FirstRootMotionTimeCode = UsdTimeSamples[0];
                    LastRootMotionTimeCode = UsdTimeSamples[UsdTimeSamples.size() - 1];
                }
            }
        }
    }
    ED_COUT << " RootMotionTimeCode " << FirstRootMotionTimeCode << " " << LastRootMotionTimeCode << "\n";
    ED_COUT << " RootMotionPrimPath " << RootMotionPrim.GetPath() << "\n";

    const double StageStartTimeCode = FirstJointSampleTimeCode;
    const double StageEndTimeCode = LastJointSampleTimeCode;

    // TODO Min and max time code compare for All of First and Last.
    const double StageStartSeconds = StageStartTimeCode / StageTimeCodesPerSecond;
    const double StageSequenceLengthTimeCodes = StageEndTimeCode - StageStartTimeCode;
    const double LayerSequenceLengthTimeCodes = StageSequenceLengthTimeCodes / Offset.GetScale();
    const double LayerSequenceLengthSeconds = std::max<double>(LayerSequenceLengthTimeCodes / LayerTimeCodesPerSecond, ED_MINIMUM_ANIMATION_LENGTH);
    const double LayerStartTimeCode = (StageStartTimeCode - Offset.GetOffset()) / Offset.GetScale();
    const double LayerStartSeconds = LayerStartTimeCode / LayerTimeCodesPerSecond;
    ED_COUT << " Times " << StageStartSeconds << " " << StageSequenceLengthTimeCodes << " " << LayerSequenceLengthTimeCodes << " " << LayerSequenceLengthSeconds << " " << LayerStartTimeCode << " " << LayerStartSeconds << " " << StageStartTimeCode << " " << LayerTimeCodesPerSecond << "\n";

    // Just bake each time code in the source layer as a frame
    const int32 NumBakedFrames = std::round(std::max(LayerSequenceLengthSeconds * LayerTimeCodesPerSecond + 1.0, 1.0));
    const double StageBakeIntervalTimeCodes = 1.0 * Offset.GetScale();
    ED_COUT << " Times " << NumBakedFrames << " " << StageBakeIntervalTimeCodes << "\n";

    EAnimationInfo AnimInfo;
    AnimInfo.NumBakedFrames = NumBakedFrames;
    AnimInfo.StageStartTimeCode = StageStartTimeCode;
    AnimInfo.StageTimeCodesPerSecond = StageTimeCodesPerSecond;
    AnimInfo.StageBakeIntervalTimeCodes = StageBakeIntervalTimeCodes;
    AnimInfo.StageSequenceLengthTimeCodes = StageSequenceLengthTimeCodes;

    OutAnimData.AnimationInfo = AnimInfo;

    // Bake the animation for each frame.
    int32 NumBones = InSkeletonBones.size();
    ED_COUT << " Num Bones " << NumBones << "\n";
    if (NumJointTransformSamples >= 2){
        pxr::VtArray<ERawAnimSequenceTrack> JointTracks;
        JointTracks.resize(NumBones);

        for (int32 BoneIndex = 0; BoneIndex < NumBones; ++BoneIndex)
        {
            ERawAnimSequenceTrack& JointTrack = JointTracks[BoneIndex];
            JointTrack.PosKeys.reserve(NumBakedFrames);
            JointTrack.RotKeys.reserve(NumBakedFrames);
            JointTrack.ScaleKeys.reserve(NumBakedFrames);
        }

        ETransform RootMotionTransform;

        pxr::UsdGeomXformCache XformCache{};

        pxr::VtArray<pxr::GfMatrix4d> UsdJointLocalTransforms;
        pxr::VtArray<pxr::GfMatrix4d> UsdJointWorldTransforms;
        pxr::VtArray<pxr::GfMatrix4d> UsdJointWorldRestTransforms;
        pxr::VtArray<pxr::GfMatrix4d> UsdJointRestRelativeTransforms;
        pxr::VtArray<pxr::GfMatrix4d> UsdJointWorldBindTransforms;
        pxr::VtArray<pxr::GfMatrix4d> UsdSkinningTransforms;
        pxr::VtArray<pxr::GfMatrix4d> UsdJointSkelTransforms;

        pxr::VtArray<ETransform> RootMotionTransforms;

        for (int32 FrameIndex = 0; FrameIndex < NumBakedFrames; ++FrameIndex)
        {
            bool atRest = true;
            const double StageFrameTimeCodes = StageStartTimeCode + FrameIndex * StageBakeIntervalTimeCodes;
            InUsdSkeletonQuery.ComputeJointLocalTransforms(&UsdJointLocalTransforms, StageFrameTimeCodes);

            for (int32 BoneIndex = 0; BoneIndex < NumBones; ++BoneIndex) {
                pxr::GfMatrix4d &UsdJointTransform = UsdJointLocalTransforms[BoneIndex];
                //if(FrameIndex == 0) {
                //    ED_COUT << " Local FrameIndex " << FrameIndex << " BoneIndex " << BoneIndex << " " << UsdJointTransform  << "\n";
                //}
            }
            /*
            // Joint World atRest
            InUsdSkeletonQuery.ComputeJointWorldTransforms(&UsdJointWorldRestTransforms, &XformCache, atRest);
            for (int32 BoneIndex = 0; BoneIndex < NumBones; ++BoneIndex) {
                pxr::GfMatrix4d &UsdJointTransform = UsdJointWorldRestTransforms[BoneIndex];
                if(FrameIndex == 0) {
                    ED_COUT << " World Rest FrameIndex " << FrameIndex << " BoneIndex " << BoneIndex << " " << UsdJointTransform  << "\n";
                }
            }

            // Joint World
            InUsdSkeletonQuery.ComputeJointWorldTransforms(&UsdJointWorldTransforms, &XformCache, false);
            for (int32 BoneIndex = 0; BoneIndex < NumBones; ++BoneIndex) {
                pxr::GfMatrix4d &UsdJointTransform = UsdJointWorldTransforms[BoneIndex];
                if(FrameIndex == 0) {
                    ED_COUT << " World FrameIndex " << FrameIndex << " BoneIndex " << BoneIndex << " " << UsdJointTransform  << "\n";
                }
            }

            // Joint Rest Relative
            InUsdSkeletonQuery.ComputeJointRestRelativeTransforms(&UsdJointRestRelativeTransforms, StageFrameTimeCodes);
            for (int32 BoneIndex = 0; BoneIndex < NumBones; ++BoneIndex) {
                pxr::GfMatrix4d &UsdJointTransform = UsdJointRestRelativeTransforms[BoneIndex];
                if(FrameIndex == 0) {
                    ED_COUT << " RestRelative FrameIndex " << FrameIndex << " BoneIndex " << BoneIndex << " " << UsdJointTransform  << "\n";
                }
            }

            // World Bind
            InUsdSkeletonQuery.GetJointWorldBindTransforms(&UsdJointWorldBindTransforms);
            for (int32 BoneIndex = 0; BoneIndex < NumBones; ++BoneIndex) {
                pxr::GfMatrix4d &UsdJointTransform = UsdJointWorldBindTransforms[BoneIndex];
                if(FrameIndex == 0) {
                    ED_COUT << " World Bind FrameIndex " << FrameIndex << " BoneIndex " << BoneIndex << " " << UsdJointTransform  << "\n";
                }
            }

            // Skinning
            InUsdSkeletonQuery.ComputeSkinningTransforms(&UsdSkinningTransforms, StageFrameTimeCodes);
            for (int32 BoneIndex = 0; BoneIndex < NumBones; ++BoneIndex) {
                pxr::GfMatrix4d &UsdJointTransform = UsdSkinningTransforms[BoneIndex];
                if(FrameIndex == 0) {
                    ED_COUT << " Skinning FrameIndex " << FrameIndex << " BoneIndex " << BoneIndex << " " << UsdJointTransform  << "\n";
                }
            }
            */

            {
                bool* OutResetTransformStack = nullptr;
                const bool bSuccess = ConvertXformable(
                        Stage,
                        RootMotionXformable,
                        RootMotionTransform,
                        StageFrameTimeCodes,
                        OutResetTransformStack
                );
            }

            RootMotionTransforms.emplace_back(RootMotionTransform);

            // Joint Skel
            InUsdSkeletonQuery.ComputeJointSkelTransforms(&UsdJointSkelTransforms, StageFrameTimeCodes);
            for (int32 BoneIndex = 0; BoneIndex < NumBones; ++BoneIndex) {
                pxr::GfMatrix4d &UsdJointTransform = UsdJointSkelTransforms[BoneIndex];
                ETransform ConvertedJointTransform = ConvertMatrix(StageInfo, UsdJointTransform);
                //if(FrameIndex == 0) {
                //    ED_COUT << " Joint Skel FrameIndex " << FrameIndex << " BoneIndex " << BoneIndex << " " << UsdJointTransform  << "\n";
                //}

                ERawAnimSequenceTrack& JointTrack = JointTracks[BoneIndex];
                JointTrack.PosKeys.emplace_back(ConvertedJointTransform.GetTranslation());
                JointTrack.RotKeys.emplace_back(ConvertedJointTransform.GetRotation());
                JointTrack.ScaleKeys.emplace_back(ConvertedJointTransform.GetScale3D());
                JointTrack.ConvertedTransforms.emplace_back(ConvertedJointTransform);
                JointTrack.UsdTransforms.emplace_back(UsdJointTransform);
            }

            /*
            for (int32 BoneIndex = 0; BoneIndex < NumBones; ++BoneIndex)
            {
                pxr::GfMatrix4d& UsdJointTransform = UsdJointLocalTransforms[BoneIndex];
                ETransform UEJointTransform = ConvertMatrix(StageInfo, UsdJointTransform);

                // Concatenate the root bone transform with the transform track actually present on the skel root as a whole
                if (BoneIndex == 0)
                {
                    // We don't care about resetXformStack here: We'll always use the root motion prim's transform as
                    // a local transformation anyway
                    bool* OutResetTransformStack = nullptr;
                    const bool bSuccess = ConvertXformable(
                            Stage,
                            RootMotionXformable,
                            RootMotionTransform,
                            StageFrameTimeCodes,
                            OutResetTransformStack
                    );

                    if(FrameIndex == 0) {
                        ED_COUT << " Local FrameIndex " << FrameIndex << " BoneIndex " << BoneIndex << " " << UsdJointTransform  << "\n";
                        ED_COUT << " - - Root MotionTransform: " << RootMotionTransform << "\n";
                    }

                    if (bSuccess)
                    {
                        UEJointTransform = UEJointTransform * RootMotionTransform;
                    }

                    //ED_COUT << " - - Joint Transform: " << UEJointTransform << "\n";
                }

                ERawAnimSequenceTrack& JointTrack = JointTracks[BoneIndex];
                JointTrack.PosKeys.emplace_back(UEJointTransform.GetTranslation());
                JointTrack.RotKeys.emplace_back(UEJointTransform.GetRotation());
                JointTrack.ScaleKeys.emplace_back(UEJointTransform.GetScale3D());
                JointTrack.Transforms.emplace_back(UEJointTransform);
            }
             */
        }

        OutAnimData.JointAnimTracks = JointTracks;
        OutAnimData.RootMotionTransforms = RootMotionTransforms;

        ED_COUT << " Joint Tracks " << JointTracks << "\n";
    }

    // Add float tracks to animate morph target weights
    if (InBlendShapes && InSkinningTargets){
        pxr::UsdSkelAnimQuery UsdAnimQuery = AnimQuery;

        pxr::VtTokenArray SkelAnimChannelOrder = UsdAnimQuery.GetBlendShapeOrder();
        int32 NumSkelAnimChannels = SkelAnimChannelOrder.size();

        OutAnimData.BlendShapeChannelOrder = SkelAnimChannelOrder;
        OutAnimData.BlendShapeWeights.resize(NumSkelAnimChannels);
        for(auto& BlendShape: OutAnimData.BlendShapeWeights){
            BlendShape.resize(NumBakedFrames);
        }

        ED_COUT << " SkelAnimChannelOrder Size " << NumSkelAnimChannels << " " << SkelAnimChannelOrder << "\n";
        if (NumSkelAnimChannels > 0)
        {
            pxr::VtArray<float> WeightsForFrame;
            for (int32 FrameIndex = 0; FrameIndex < NumBakedFrames; ++FrameIndex)
            {
                const double StageFrameTimeCodes = StageStartTimeCode + FrameIndex * StageBakeIntervalTimeCodes;
                const double LayerFrameTimeCodes = (StageFrameTimeCodes - Offset.GetOffset()) / Offset.GetScale();
                const double LayerFrameSeconds = LayerFrameTimeCodes / LayerTimeCodesPerSecond - LayerStartSeconds;

                UsdAnimQuery.ComputeBlendShapeWeights(&WeightsForFrame, pxr::UsdTimeCode(StageFrameTimeCodes));

                for (int32 SkelAnimChannelIndex = 0; SkelAnimChannelIndex < NumSkelAnimChannels; ++SkelAnimChannelIndex)
                {
                    float Weight = WeightsForFrame[SkelAnimChannelIndex];
                    //ED_COUT << " FrameIndex " << FrameIndex << " SkelAnimChannelIndex " << SkelAnimChannelIndex << "\n";
                    //ED_COUT << " LayerFrameSeconds " << LayerFrameSeconds << " Weight " << Weight << "\n";

                    OutAnimData.BlendShapeWeights[SkelAnimChannelIndex][FrameIndex] = Weight;
                }
            }

            pxr::VtArray<pxr::SdfPath> PathsToSkinnedPrims;
            for (const pxr::UsdSkelSkinningQuery& SkinningQuery : *InSkinningTargets)
            {
                // In USD, the skinning target need not be a mesh
                if (pxr::UsdGeomMesh SkinningMesh = pxr::UsdGeomMesh(SkinningQuery.GetPrim()))
                {
                    PathsToSkinnedPrims.emplace_back(SkinningMesh.GetPrim().GetPath());
                }
            }

            ED_COUT << "Repeated Elements " << Helper::HasRepeatedElements(PathsToSkinnedPrims) << "\n";

            OutAnimData.BlendShapePathsToSkinnedPrims = PathsToSkinnedPrims;

            //ED_COUT << " Paths To Skinned Prims " << PathsToSkinnedPrims << "\n";
            // TODO Interpreted (Variants)
        }
    }
    //ED_COUT << "BlendShape Weights: " << OutAnimData.BlendShapeWeights << "\n";
    return true;
}

pxr::SdfLayerOffset GetPrimToStageOffset(const pxr::UsdPrim& Prim)
{
    // In most cases all we care about is an offset from the prim's layer to the stage, but it is also possible for a prim
    // to directly reference another layer with an offset and scale as well, and this function will pick up on that. Example:
    //
    // def SkelRoot "Model" (
    //	  prepend references = @sublayer.usda@ ( offset = 15; scale = 2.0 )
    // )
    // {
    // }
    //
    // Otherwise, this function really has the same effect as GetLayerToStageOffset, but we need to use an actual prim to be able
    // to get USD to combine layer offsets and scales for us (via UsdPrimCompositionQuery).

    pxr::SdfLayerRefPtr StrongestLayerForPrim = FindLayerForPrim(Prim);
    pxr::UsdPrim UsdPrim{Prim};
    pxr::UsdPrimCompositionQuery PrimCompositionQuery(UsdPrim);
    pxr::UsdPrimCompositionQuery::Filter Filter;
    Filter.hasSpecsFilter = pxr::UsdPrimCompositionQuery::HasSpecsFilter::HasSpecs;
    PrimCompositionQuery.SetFilter(Filter);

    for (const pxr::UsdPrimCompositionQueryArc& CompositionArc : PrimCompositionQuery.GetCompositionArcs())
    {
        if (pxr::PcpNodeRef Node = CompositionArc.GetTargetNode())
        {
            ED_COUT << " CompositionArc Node Path " << Node.GetPath() << "\n";
            pxr::SdfLayerOffset Offset;

            // This part of the offset will handle direct prim references
            const pxr::PcpMapExpression& MapToRoot = Node.GetMapToRoot();
            if (!MapToRoot.IsNull())
                Offset = MapToRoot.GetTimeOffset();

            if (const pxr::SdfLayerOffset* LayerOffset = Node.GetLayerStack()->GetLayerOffsetForLayer(pxr::SdfLayerRefPtr{StrongestLayerForPrim}))
                Offset = Offset * (*LayerOffset);

            ED_COUT << " Offset " << Offset.GetOffset() << " Scale " << Offset.GetScale() << "\n";
            return pxr::SdfLayerOffset{Offset.GetOffset(), Offset.GetScale()};
        }
    }

    return pxr::SdfLayerOffset{};
}

pxr::SdfLayerRefPtr FindLayerForPrim(const pxr::UsdPrim& Prim)
{
    if (!Prim)
        return {};

    // Use this instead of UsdPrimCompositionQuery as that one can simply fail in some scenarios
    // (e.g. empty parent layer pointing at a sublayer with a prim, where it fails to provide the sublayer arc's layer)
    for (const pxr::SdfPrimSpecHandle& Handle : Prim.GetPrimStack())
    {
        if (Handle)
        {
            if (pxr::SdfLayerHandle Layer = Handle->GetLayer())
            {
                ED_COUT << " Find Layer For Prim " << Layer->GetDisplayName() << "\n";
                return Layer;
            }
        }
    }

    return Prim.GetStage()->GetRootLayer();
}