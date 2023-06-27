#include <zeno/zeno.h>
#include <zeno/logger.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/unreal/UnrealTool.h>
#include <zeno/types/DictObject.h>
#include <zeno/funcs/PrimitiveUtils.h>

namespace zeno {

// Landscape input node
struct RemoteLandscapeInput : public INode {
    void apply() override {
        std::string SubjectName = get_input2<std::string>("SubjectName");
        std::optional<remote::HeightField> Data = remote::StaticRegistry.Get<remote::HeightField>(SubjectName, zeno::remote::StaticFlags.GetCurrentSession(), true);
        if (Data.has_value()) {
            const float ScaleX = Data->LandscapeScaleX == .0f ? 100.f : Data->LandscapeScaleX;
            const float ScaleY = Data->LandscapeScaleY == .0f ? 100.f : Data->LandscapeScaleY;
            const float ScaleZ = Data->LandscapeScaleZ == .0f ? 100.f : Data->LandscapeScaleZ;
//            const float Scale = 1.0f;
            std::shared_ptr<zeno::PrimitiveObject> Prim = remote::ConvertHeightDataToPrimitiveObject(Data.value(), 0, 0, {ScaleX, ScaleY, ScaleZ});
            set_output2("prim", Prim);
        } else {
            zeno::log_error("landscape data not found.");
        }
    }
};

ZENO_DEFNODE(RemoteLandscapeInput)({
    {
        {"string", "SubjectName", "ReservedName_Landscape"},
    },
    {
        {"prim"},
    },
    {},
    {"Unreal"},
});

struct RemoteSurfaceSampleInput : public INode {
    void apply() override {
        std::string SubjectName = get_input2<std::string>("SubjectName");
        std::optional<remote::PointSet> Data = remote::StaticRegistry.Get<remote::PointSet>(SubjectName, zeno::remote::StaticFlags.GetCurrentSession(), true);
        if (Data.has_value()) {
        } else {
            zeno::log_error("surface sample data not found.");
        }
    }
};

ZENO_DEFNODE(RemoteSurfaceSampleInput)({
    {},
    {
        {"prim"},
    },
    {},
    {"Unreal"},
});

struct CreateMetaData : public INode {
    void apply() override {
        auto dict = get_input2<zeno::DictObject>("dict");
        auto meta = std::make_shared<zeno::remote::MetaData>();
        if (dict) {
            const auto Literial = dict->getLiterial<std::string>();
            for (auto& Item : Literial) {
                meta->Data.insert_or_assign(Item.first, Item.second);
            }
        }
        set_output2("meta", std::move(meta));
    }
};

ZENO_DEFNODE(CreateMetaData)({
    {
        {"dict", "dict" },
    },
    {
        {"meta"},
    },
    {},
    {"Unreal"},
});

struct PackVec3fToString : public INode {
    void apply() override {
        auto vec3f = get_input2<zeno::vec3f>("vec3f");
        std::string str = std::to_string(vec3f.at(0)) + "," + std::to_string(vec3f.at(1)) + "," + std::to_string(vec3f.at(2));
        set_output2("string", str);
    }
};

ZENO_DEFNODE(PackVec3fToString)({
    {
        { "vec3f", "vec3f" },
    },
    {
        {"string", "string"},
    },
    {},
    {"Unreal"},
});

struct PackBoundDiffToString : public INode {
    void apply() override {
        auto OriginPrim = get_input2<zeno::PrimitiveObject>("OriginPrim");
        auto CurrentPrim = get_input2<zeno::PrimitiveObject>("CurrentPrim");

        auto [OriginMin, OriginMax] = zeno::primBoundingBox(OriginPrim.get());
        auto [CurrentMin, CurrentMax] = zeno::primBoundingBox(CurrentPrim.get());
        auto OriginEdgeLength = (OriginMax - OriginMin);
        float PercentTop = (CurrentMax.at(0) - OriginMax.at(0)) / OriginEdgeLength.at(0);
        float PercentBottom = (CurrentMin.at(0) - OriginMin.at(0)) / OriginEdgeLength.at(0);
        float PercentLeft = (CurrentMax.at(2) - OriginMax.at(2)) / OriginEdgeLength.at(2);
        float PercentRight = (CurrentMin.at(2) - OriginMin.at(2)) / OriginEdgeLength.at(2);

        std::string str = std::to_string(PercentTop) + "," + std::to_string(PercentBottom) + "," + std::to_string(PercentLeft) + "," + std::to_string(PercentRight);
        set_output2("string", str);
    }
};

ZENO_DEFNODE(PackBoundDiffToString)({
    {
        { "prim", "OriginPrim" },
        { "prim", "CurrentPrim" },
    },
    {
        {"string", "string"},
    },
    {},
    {"Unreal"},
});

}
