#ifndef YE_USDHELPER_H
#define YE_USDHELPER_H

#include "usdDefinition.h"

#include <pxr/pxr.h>
#include <pxr/base/vt/types.h>
#include <pxr/base/gf/matrix4f.h>

namespace EAxis
{
    enum Type
    {
        None,
        X,
        Y,
        Z,
    };
}

template<typename Enum>
constexpr bool EnumHasAllFlags(Enum Flags, Enum Contains)
{
    return (((__underlying_type(Enum))Flags) & (__underlying_type(Enum))Contains) == ((__underlying_type(Enum))Contains);
}

template<typename Enum>
constexpr bool EnumHasAnyFlags(Enum Flags, Enum Contains)
{
    return ((( __underlying_type(Enum))Flags) & (__underlying_type(Enum))Contains) != 0;
}

template<typename Enum>
void EnumAddFlags(Enum& Flags, Enum FlagsToAdd)
{
    Flags |= FlagsToAdd;
}

template<typename Enum>
void EnumRemoveFlags(Enum& Flags, Enum FlagsToRemove)
{
    Flags &= ~FlagsToRemove;
}

bool IsNearlyEqual(double A, double B, double ErrorTolerance = 1.e-8);

bool IsNearlyZero(float Value, float ErrorTolerance = 1.e-8);

inline float Sqrt(float Value) { return sqrtf(Value); }

inline double Sqrt(double Value) { return sqrt(Value); }

template <typename T> struct TRemoveReference      { typedef T Type; };
template <typename T> struct TRemoveReference<T& > { typedef T Type; };
template <typename T> struct TRemoveReference<T&&> { typedef T Type; };

template <typename T>
inline typename TRemoveReference<T>::Type&& MoveTemp(T&& Obj)
{
    typedef typename TRemoveReference<T>::Type CastType;
    return (CastType&&)Obj;
}

template <typename T>
inline void Swap(T& A, T& B)
{
    T Temp = MoveTemp(A);
    A = MoveTemp(B);
    B = MoveTemp(Temp);
}

template <typename T>
inline pxr::GfVec3f ExtractScaling(pxr::GfMatrix4f M){
    pxr::GfVec3f Scale3D(0,0,0);

    // FIXEM Too big?
    float Tolerance = 0.001f;

    float SquareSum0 = (M[0][0] * M[0][0]) + (M[0][1] * M[0][1]) + (M[0][2] * M[0][2]);
    float SquareSum1 = (M[1][0] * M[1][0]) + (M[1][1] * M[1][1]) + (M[1][2] * M[1][2]);
    float SquareSum2 = (M[2][0] * M[2][0]) + (M[2][1] * M[2][1]) + (M[2][2] * M[2][2]);

    //ED_COUT << "      Extract Scaling " << SquareSum0 << " " <<SquareSum1 << " "<<SquareSum2<<"\n";

    if (SquareSum0 > Tolerance)
    {
        T Scale0 = Sqrt(SquareSum0);
        Scale3D[0] = Scale0;
        T InvScale0 = 1.f / Scale0;
        M[0][0] *= InvScale0;
        M[0][1] *= InvScale0;
        M[0][2] *= InvScale0;
    }
    else
        Scale3D[0] = 0;

    if (SquareSum1 > Tolerance)
    {
        T Scale1 = Sqrt(SquareSum1);
        Scale3D[1] = Scale1;
        T InvScale1 = 1.f / Scale1;
        M[1][0] *= InvScale1;
        M[1][1] *= InvScale1;
        M[1][2] *= InvScale1;
    }
    else
        Scale3D[1] = 0;

    if (SquareSum2 > Tolerance)
    {
        T Scale2 = Sqrt(SquareSum2);
        Scale3D[2] = Scale2;
        T InvScale2 = 1.f / Scale2;
        M[2][0] *= InvScale2;
        M[2][1] *= InvScale2;
        M[2][2] *= InvScale2;
    }
    else
        Scale3D[2] = 0;

    //ED_COUT << "      Extract Scaling " << Scale3D << "\n";
    return Scale3D;
}

inline void SetAxis(pxr::GfMatrix4f& M, int i, pxr::GfVec3f Axis){
    M[i][0] = Axis[0];
    M[i][1] = Axis[1];
    M[i][2] = Axis[2];
}

template<typename T>
inline pxr::GfVec3f GetScaledAxis(pxr::GfMatrix4f& M, EAxis::Type InAxis)
{
    switch (InAxis)
    {
        case EAxis::X:
            return {M[0][0], M[0][1], M[0][2]};

        case EAxis::Y:
            return {M[1][0], M[1][1], M[1][2]};

        case EAxis::Z:
            return {M[2][0], M[2][1], M[2][2]};

        default:
            return {0, 0, 0};
    }
}

inline pxr::GfVec3f GetOrigin(pxr::GfMatrix4f& M){
    return {M[3][0], M[3][1], M[3][2]};
}

template<class T>
inline static constexpr T Clamp(const T X, const T Min, const T Max)
{
    return (X < Min) ? Min : (X < Max) ? X : Max;
}

inline std::vector<std::string> SplitStringByDelimiter(const std::string& str, const std::string& delimiter) {
    std::vector<std::string> substrings;
    size_t start = 0, end = 0;
    while ((end = str.find(delimiter, start)) != std::string::npos) {
        substrings.push_back(str.substr(start, end - start));
        start = end + delimiter.length();
    }
    substrings.push_back(str.substr(start));
    return substrings;
}

namespace Helper {
/*
 * e.g.
*       // Compute Bone Global Transform
        pxr::VtArray<ETransform> world_skel_transform{};
        auto size_of_bones = skel_data.SkeletonBones.size();
        world_skel_transform.resize(size_of_bones);
        for(int i=0; i<size_of_bones; ++i){
            auto& skelbone = skel_data.SkeletonBones[i];
            world_skel_transform[i] = Helper::RecursiveGetBoneTrans(skelbone, skel_data.SkeletonBones, ETransform());
        }
 */
#define RecursiveGetTransParams EBone& Bone, pxr::VtArray<EBone>& Bones, ETransform ParentTrans
    // Skel Mesh Transform
    inline auto RecursiveGetBoneTrans = [](RecursiveGetTransParams) -> ETransform {
        std::function<ETransform(RecursiveGetTransParams)> Recursivefunc;
        Recursivefunc = [&](RecursiveGetTransParams) -> ETransform {
            auto NumChild = Bone.NumChildren;
            auto ParentIndex = Bone.ParentIndex;
            auto LocalTrans = Bone.BonePos.Transform;
            auto AppliedTrans = ParentTrans * LocalTrans;
            if (ParentIndex >= 0) {
                auto &ParentBone = Bones[ParentIndex];
                return Recursivefunc(ParentBone, Bones, AppliedTrans);
            }
            return AppliedTrans;
        };
        return Recursivefunc(Bone, Bones, ParentTrans);
    };

    template<typename T>
    bool HasRepeatedElements(const pxr::VtArray<T>& vec) {
        std::set<T> uniqueSet;
        for (const auto& elem : vec) {
            if (uniqueSet.find(elem) != uniqueSet.end()) {
                return true;
            }
            uniqueSet.insert(elem);
        }
        return false;
    }

    template <typename T>
    inline int GetElemIndex(pxr::VtArray<T> v, T K)
    {
        auto it = std::find(v.begin(), v.end(), K);

        // If element was found
        if (it != v.end())
        {
            // calculating the index of K
            int index = it - v.begin();
            return index;
        }
        else {
            return -1;
        }
    }

    template <typename T>
    int GetImportedAnimMeshDataKey(int Frame, std::map<int, T>& MapData){
        if(MapData.find(Frame) != MapData.end()){
            return Frame;
        }else{
            std::vector<int> keys;
            typename std::map<int, T>::iterator it;
            for (it=MapData.begin(); it!=MapData.end(); ++it) {
                keys.push_back(it->first);
            }
            std::sort(keys.begin(), keys.end());
            for(int key : keys){
                if(Frame <= key){
                    return key;
                }
            }
            return keys.back();
        }
    }

    template <typename T>
    std::pair<int, int> GetImportedAnimMeshRange(std::map<int, T>& MapData){
        std::vector<int> keys;
        typename std::map<int, T>::iterator it;
        for (it=MapData.begin(); it!=MapData.end(); ++it) {
            keys.push_back(it->first);
        }
        std::sort(keys.begin(), keys.end());
        std::pair<int, int> range;
        range.first = keys.front();
        range.second = keys.back();

        return range;
    }
}

namespace Helper{
    pxr::GfMatrix4f ETransformToUsdMatrix(ETransform& Transform);
    double GetRangeFrameKey(double Frame, EAnimationInfo& AnimInfo);
    bool EvalSkeletalSkin(ESkelImportData& SkelImportData,
                          ESkeletalMeshImportData& SkelMeshImportData,
                          pxr::VtArray<pxr::GfVec3f>& MeshPoints,
                          int32 FrameIndex,
                          pxr::VtArray<pxr::GfVec3f>& OutSkinPoints);
    bool EvalSkeletalBlendShape(ESkelImportData& SkelImportData,
                                ESkeletalMeshImportData& SkelMeshImportData,
                                int32 FrameIndex,
                                pxr::VtArray<pxr::GfVec3f>& OutDeformPoints);
}

#endif //YE_USDHELPER_H
