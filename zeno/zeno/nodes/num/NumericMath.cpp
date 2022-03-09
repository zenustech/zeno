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
        auto normal = get_input<NumericObject>("normal")->get<vec3f>();
        normal = normalize(normal);
        vec3f tangent, bitangent;
        if (has_input("tangent")) {
            tangent = get_input<NumericObject>("tangent")->get<vec3f>();
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

        set_output("normal", std::make_shared<NumericObject>(normal));
        set_output("tangent", std::make_shared<NumericObject>(tangent));
        set_output("bitangent", std::make_shared<NumericObject>(bitangent));
    }
};

ZENDEFNODE(MakeOrthonormalBase, {
    {{"vec3f", "normal", "0,0,1"}, {"vec3f", "tangent", "0,1,0"}},
    {{"vec3f", "normal"}, {"vec3f", "tangent"}, {"vec3f", "bitangent"}},
    {},
    {"math"},
});


struct OrthonormalBase : INode {
    virtual void apply() override {
        std::unique_ptr<orthonormal> orb;

        auto normal = get_input<NumericObject>("normal")->get<vec3f>();
        if (has_input("tangent")) {
            auto tangent = get_input<NumericObject>("tangent")->get<vec3f>();
            orb = std::make_unique<orthonormal>(normal, tangent);
        } else {
            orb = std::make_unique<orthonormal>(normal);
        }

        set_output("normal", std::make_shared<NumericObject>(orb->normal));
        set_output("tangent", std::make_shared<NumericObject>(orb->tangent));
        set_output("bitangent", std::make_shared<NumericObject>(orb->bitangent));
    }
};

ZENDEFNODE(OrthonormalBase, {
    {{"vec3f", "normal", "0,0,1"}, {"vec3f", "tangent", "0,1,0"}},
    {{"vec3f", "normal"}, {"vec3f", "tangent"}, {"vec3f", "bitangent"}},
    {},
    {"math"},
});


struct AABBCollideDetect : INode {
    virtual void apply() override {
        auto bminA = get_input<NumericObject>("bminA")->get<vec3f>();
        auto bmaxA = get_input<NumericObject>("bmaxA")->get<vec3f>();
        auto bminB = get_input<NumericObject>("bminB")->get<vec3f>();
        auto bmaxB = get_input<NumericObject>("bmaxB")->get<vec3f>();

        // https://www.cnblogs.com/liez/p/11965027.html
        bool overlap = alltrue(abs(bminA + bmaxA - bminB - bmaxB) <= (bmaxA - bminA + bmaxB - bminB));
        set_output2("overlap", overlap);
        bool AinsideB = alltrue(bminA >= bminB && bmaxA <= bmaxB);
        set_output2("AinsideB", AinsideB);
        bool BinsideA = alltrue(bminA <= bminB && bmaxA >= bmaxB);
        set_output2("BinsideA", BinsideA);
    }
};

ZENDEFNODE(AABBCollideDetect, {
    {{"vec3f", "bminA"}, {"vec3f", "bmaxA"}, {"vec3f", "bminB"}, {"vec3f", "bmaxB"}},
    {{"bool", "overlap"}, {"bool", "AinsideB"}, {"bool", "BinsideA"}},
    {},
    {"math"},
});

struct ProjectAndNormalize : INode {
    virtual void apply() override {
        auto vec = get_input<NumericObject>("vec")->get<vec3f>();
        auto plane = get_input2<std::string>("plane");

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
        direction *= get_input2<float>("directionScale");
        length *= get_input2<float>("lengthScale");
        height *= get_input2<float>("heightScale");
        height += get_input2<float>("heightOffset");

        //log_info("length: {}", length);
        //log_info("direction: {} {} {}", direction[0], direction[1], direction[2]);

        set_output("direction", std::make_shared<NumericObject>(direction));
        set_output("length", std::make_shared<NumericObject>(length));
        set_output("height", std::make_shared<NumericObject>(height));
        set_output("phase", std::make_shared<NumericObject>(phase));
    }
};

ZENDEFNODE(ProjectAndNormalize, {
    {
    {"vec3f", "vec"},
    {"enum XY YX YZ ZY ZX XZ", "plane", "XY"},
    {"float", "directionScale", "1"},
    {"float", "lengthScale", "1"},
    {"float", "heightScale", "1"},
    {"float", "heightOffset", "0"},
    },
    {
    {"vec3f", "direction"},
    {"float", "length"},
    {"float", "height"},
    {"float", "phase"},
    },
    {},
    {"math"},
});

struct CalcDirectionFromAngle : INode {
    virtual void apply() override {
        auto plane = get_input2<std::string>("plane");

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

        auto angle = get_input<NumericObject>("angle")->get<float>();
        auto length = get_input<NumericObject>("length")->get<float>();
        angle *= M_PI / 180;

        auto orb0v = std::cos(angle) * length;
        auto orb1v = std::sin(angle) * length;
        auto direction = orb[0] * orb0v + orb[1] * orb1v;

        set_output("direction", std::make_shared<NumericObject>(direction));
    }
};

ZENDEFNODE(CalcDirectionFromAngle, {
    {
    {"float", "angle", "0"},
    {"enum XY YX YZ ZY ZX XZ", "plane", "XY"},
    {"float", "length", "1"},
    },
    {
    {"vec3f", "direction"},
    },
    {},
    {"math"},
});

}
}
