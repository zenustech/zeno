#include <zeno/zeno.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/orthonormal.h>
//#include <zeno/utils/logger.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace zeno {
namespace {

struct MakeOrthonormalBase : INode {
    virtual void apply() override {
        auto normal = ZImpl(get_input<NumericObject>("normal"))->get<zeno::vec3f>();
        normal = normalize(normal);
        vec3f tangent, bitangent;
        if (ZImpl(has_input("tangent"))) {
            tangent = ZImpl(get_input<NumericObject>("tangent"))->get<zeno::vec3f>();
            bitangent = cross(normal, tangent);
        } else {
            tangent = vec3f(0, 0, 1);
            bitangent = cross(normal, tangent);
            if (dot(bitangent, bitangent) < 1e-5) {
                tangent = vec3f(0, 1, 0);
               bitangent = cross(normal, tangent);
            }
        }
        bitangent = normalize(bitangent);
        tangent = cross(bitangent, normal);

        ZImpl(set_output("normal", std::make_unique<NumericObject>(normal)));
        ZImpl(set_output("tangent", std::make_unique<NumericObject>(tangent)));
        ZImpl(set_output("bitangent", std::make_unique<NumericObject>(bitangent)));
    }
};

ZENDEFNODE(MakeOrthonormalBase, {
    {{gParamType_Vec3f, "normal", "0,0,1"}, {gParamType_Vec3f, "tangent", "0,1,0"}},
    {{gParamType_Vec3f, "normal"}, {gParamType_Vec3f, "tangent"}, {gParamType_Vec3f, "bitangent"}},
    {},
    {"math"},
});


struct OrthonormalBase : INode {
    virtual void apply() override {
        std::unique_ptr<orthonormal> orb;

        auto normal = ZImpl(get_input<NumericObject>("normal"))->get<zeno::vec3f>();
        if (ZImpl(has_input("tangent"))) {
            auto tangent = ZImpl(get_input<NumericObject>("tangent"))->get<zeno::vec3f>();
            orb = std::make_unique<orthonormal>(normal, tangent);
        } else {
            orb = std::make_unique<orthonormal>(normal);
        }

        ZImpl(set_output("normal", std::make_unique<NumericObject>(orb->normal)));
        ZImpl(set_output("tangent", std::make_unique<NumericObject>(orb->tangent)));
        ZImpl(set_output("bitangent", std::make_unique<NumericObject>(orb->bitangent)));
    }
};

ZENDEFNODE(OrthonormalBase, {
    {{gParamType_Vec3f, "normal", "0,0,1"}, {gParamType_Vec3f, "tangent", "0,1,0"}},
    {{gParamType_Vec3f, "normal"}, {gParamType_Vec3f, "tangent"}, {gParamType_Vec3f, "bitangent"}},
    {},
    {"math"},
});


struct PixarOrthonormalBase : INode {
    virtual void apply() override {
        auto normal = ZImpl(get_input<NumericObject>("normal"))->get<zeno::vec3f>();
        vec3f tangent{}, bitangent{};
        if (ZImpl(has_input("tangent"))) {
            tangent = ZImpl(get_input<NumericObject>("tangent"))->get<zeno::vec3f>();
            guidedPixarONB(normal, tangent, bitangent);
        } else {
            pixarONB(normal, tangent, bitangent);
        }

        ZImpl(set_output("normal", std::make_unique<NumericObject>(normal)));
        ZImpl(set_output("tangent", std::make_unique<NumericObject>(tangent)));
        ZImpl(set_output("bitangent", std::make_unique<NumericObject>(bitangent)));
    }
};

ZENDEFNODE(PixarOrthonormalBase, {
    {{gParamType_Vec3f, "normal", "0,0,1"}, {gParamType_Vec3f, "tangent", "0,1,0"}},
    {{gParamType_Vec3f, "normal"}, {gParamType_Vec3f, "tangent"}, {gParamType_Vec3f, "bitangent"}},
    {},
    {"math"},
});


struct AABBCollideDetect : INode {
    virtual void apply() override {
        auto bminA = ZImpl(get_input<NumericObject>("bminA"))->get<zeno::vec3f>();
        auto bmaxA = ZImpl(get_input<NumericObject>("bmaxA"))->get<zeno::vec3f>();
        auto bminB = ZImpl(get_input<NumericObject>("bminB"))->get<zeno::vec3f>();
        auto bmaxB = ZImpl(get_input<NumericObject>("bmaxB"))->get<zeno::vec3f>();

        // https://www.cnblogs.com/liez/p/11965027.html
        bool overlap = alltrue(abs(bminA + bmaxA - bminB - bmaxB) <= (bmaxA - bminA + bmaxB - bminB));
        ZImpl(set_output2("overlap", overlap));
        bool AinsideB = alltrue(bminA >= bminB && bmaxA <= bmaxB);
        ZImpl(set_output2("AinsideB", AinsideB));
        bool BinsideA = alltrue(bminA <= bminB && bmaxA >= bmaxB);
        ZImpl(set_output2("BinsideA", BinsideA));
    }
};

ZENDEFNODE(AABBCollideDetect, {
    {{gParamType_Vec3f, "bminA"}, {gParamType_Vec3f, "bmaxA"}, {gParamType_Vec3f, "bminB"}, {gParamType_Vec3f, "bmaxB"}},
    {{gParamType_Bool, "overlap"}, {gParamType_Bool, "AinsideB"}, {gParamType_Bool, "BinsideA"}},
    {},
    {"math"},
});

struct ProjectAndNormalize : INode {
    virtual void apply() override {
        auto vec = ZImpl(get_input<NumericObject>("vec"))->get<zeno::vec3f>();
        auto plane = ZImpl(get_input2<std::string>("plane"));

        std::array<vec3f, 2> orb;
        vec3f X(1, 0, 0), Y(0, 1, 0), Z(0, 0, 1);
        if (plane == "XY")
            orb = {X, Y};
        else if (plane == "YX")
            orb = {Y, X};
        else if (plane == "YZ")
            orb = {Y, Z};
        else if (plane == "ZY")
            orb = {Z, Y};
        else if (plane == "ZX")
            orb = {Z, X};
        else if (plane == "XZ")
            orb = {X, Z};
        else
            throw Exception("bad plane enum: " + plane);

        auto orb0v = dot(orb[0], vec);
        auto orb1v = dot(orb[1], vec);
        auto height = dot(cross(orb[0], orb[1]), vec);
        auto phase = std::atan2(orb1v, orb0v);
        auto length = std::hypot(orb1v, orb0v);
        auto direction = orb[0] * orb0v + orb[1] * orb1v;
        if (length != 0) direction /= length;
        direction *= ZImpl(get_input2<float>("directionScale"));
        length *= ZImpl(get_input2<float>("lengthScale"));
        height *= ZImpl(get_input2<float>("heightScale"));
        height += ZImpl(get_input2<float>("heightOffset"));

        //log_info("length: {}", length);
        //log_info("direction: {} {} {}", direction[0], direction[1], direction[2]);

        ZImpl(set_output("direction", std::make_unique<NumericObject>(direction)));
        ZImpl(set_output("length", std::make_unique<NumericObject>(length)));
        ZImpl(set_output("height", std::make_unique<NumericObject>(height)));
        ZImpl(set_output("phase", std::make_unique<NumericObject>(phase)));
    }
};

ZENDEFNODE(ProjectAndNormalize, {
    {
    {gParamType_Vec3f, "vec"},
    {"enum XY YX YZ ZY ZX XZ", "plane", "XY"},
    {gParamType_Float, "directionScale", "1"},
    {gParamType_Float, "lengthScale", "1"},
    {gParamType_Float, "heightScale", "1"},
    {gParamType_Float, "heightOffset", "0"},
    },
    {
    {gParamType_Vec3f, "direction"},
    {gParamType_Float, "length"},
    {gParamType_Float, "height"},
    {gParamType_Float, "phase"},
    },
    {},
    {"math"},
});

struct CalcDirectionFromAngle : INode {
    virtual void apply() override {
        auto plane = ZImpl(get_input2<std::string>("plane"));

        std::array<vec3f, 2> orb;
        vec3f X(1, 0, 0), Y(0, 1, 0), Z(0, 0, 1);
        if (plane == "XY")
            orb = {X, Y};
        else if (plane == "YX")
            orb = {Y, X};
        else if (plane == "YZ")
            orb = {Y, Z};
        else if (plane == "ZY")
            orb = {Z, Y};
        else if (plane == "ZX")
            orb = {Z, X};
        else if (plane == "XZ")
            orb = {X, Z};
        else
            throw Exception("bad plane enum: " + plane);

        auto angle = ZImpl(get_input<NumericObject>("angle"))->get<float>();
        auto length = ZImpl(get_input<NumericObject>("length"))->get<float>();
        angle *= M_PI / 180;

        auto orb0v = std::cos(angle) * length;
        auto orb1v = std::sin(angle) * length;
        auto direction = orb[0] * orb0v + orb[1] * orb1v;

        ZImpl(set_output("direction", std::make_unique<NumericObject>(direction)));
    }
};

ZENDEFNODE(CalcDirectionFromAngle, {
    {
    {gParamType_Float, "angle", "0"},
    {"enum XY YX YZ ZY ZX XZ", "plane", "XY"},
    {gParamType_Float, "length", "1"},
    },
    {
    {gParamType_Vec3f, "direction"},
    },
    {},
    {"math"},
});


struct DegreetoRad : INode {
    virtual void apply() override {

        auto degree = ZImpl(get_input2<float>("degree"));
        float radian = degree * 0.01745329251994329576923690768489;
        ZImpl(set_output2("radian", radian));

    }
};

ZENDEFNODE(DegreetoRad, {
    {{gParamType_Float, "degree", ""}, 
    },
    {{gParamType_Float, "radian"}},
    {},
    {"math"},
});


struct RadtoDegree : INode {
    virtual void apply() override {

        auto radian = ZImpl(get_input2<float>("radian"));
        float degree = radian * 180.f / M_PI;
        ZImpl(set_output2("degree", degree));

    }
};

ZENDEFNODE(RadtoDegree, {
    {{gParamType_Float, "radian", ""}, 
    },
    {{gParamType_Float, "degree"}},
    {},
    {"math"},
});

}
}
