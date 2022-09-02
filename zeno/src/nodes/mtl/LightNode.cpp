#include <zeno/zeno.h>
#include <zeno/types/CameraObject.h>
#include <zeno/utils/arrayindex.h>

#include <zeno/extra/GlobalState.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/UserData.h>

#include <glm/mat4x4.hpp>

namespace zeno {
struct LightNode : INode {
    virtual void apply() override {
        auto isL = get_input2<int>("islight");
        auto inverdir = get_input2<int>("invertdir");
        auto position = get_input2<zeno::vec3f>("position");
        auto scale = get_input2<zeno::vec3f>("scale");
        auto rotate = get_input2<zeno::vec3f>("rotate");
        auto intensity = get_input2<float>("intensity");
        auto color = get_input2<zeno::vec3f>("color");
        auto shapeParam = get_param<std::string>("Shape");
        std::string shape;
        if (shapeParam == "Disk"){
            shape = "Disk";
        }else if(shapeParam == "Plane"){
            shape = "Plane";
        }

        auto prim = std::make_shared<zeno::PrimitiveObject>();
        auto &verts = prim->verts;
        auto &tris = prim->tris;

        // Rotate
        float ax = rotate[0] * (3.14159265358979323846 / 180.0);
        float ay = rotate[1] * (3.14159265358979323846 / 180.0);
        float az = rotate[2] * (3.14159265358979323846 / 180.0);
        glm::mat3 mx = glm::mat3(1, 0, 0, 0, cos(ax), -sin(ax), 0, sin(ax), cos(ax));
        glm::mat3 my = glm::mat3(cos(ay), 0, sin(ay), 0, 1, 0, -sin(ay), 0, cos(ay));
        glm::mat3 mz = glm::mat3(cos(az), -sin(az), 0, sin(az), cos(az), 0, 0, 0, 1);

        if(shape == "Plane"){
            auto start_point = zeno::vec3f(0.5, 0, 0.5);
            float rm = 1.0f;
            float cm = 1.0f;

            // Plane Verts
            for(int i=0; i<=1; i++){
                auto rp = start_point - zeno::vec3f(i*rm, 0, 0);
                for(int j=0; j<=1; j++){
                    auto p = rp - zeno::vec3f(0, 0, j*cm);
                    // S R T
                    p = p * scale;
                    auto gp = glm::vec3(p[0], p[1], p[2]);
                    gp = mz * my * mx * gp;
                    p = zeno::vec3f(gp.x, gp.y, gp.z);
                    auto zcp = zeno::vec3f(p[0], p[1], p[2]);
                    zcp = zcp + position;

                    verts.push_back(zcp);
                }
            }

            // Plane Indices
            tris.emplace_back(zeno::vec3i(0, 3, 1));
            tris.emplace_back(zeno::vec3i(2, 3, 0));

        }else if(shape == "Disk"){
            int divisions = 13;
            verts.emplace_back(zeno::vec3f(0, 0, 0)+position);

            for (int i = 0; i < divisions; i++) {
                float rad = 2 * 3.14159265358979323846 * i / divisions;
                auto p = zeno::vec3f(cos(rad), 0, -sin(rad));
                // S R T
                p = p * scale;
                auto gp = glm::vec3(p[0], p[1], p[2]);
                gp = mz * my * mx * gp;
                p = zeno::vec3f(gp.x, gp.y, gp.z);
                p+= position;

                verts.emplace_back(p);
                tris.emplace_back(i+1, 0, i+2);
            }
            tris[tris.size()-1] = zeno::vec3i(divisions, 0, 1);
        }

        auto &clr = prim->verts.add_attr<zeno::vec3f>("clr");
        auto c = color * intensity;
        for(int i=0; i<verts.size(); i++){
            clr[i] = c;
        }

        if(inverdir){
            for(int i=0;i<prim->tris.size(); i++){
                int tmp = prim->tris[i][2];
                prim->tris[i][2] = prim->tris[i][0];
                prim->tris[i][0] = tmp;
            }
        }

        prim->userData().setLiterial("isL", std::move(isL));
        prim->userData().setLiterial("ivD", std::move(inverdir));
        prim->userData().setLiterial("pos", std::move(position));
        prim->userData().setLiterial("scale", std::move(scale));
        prim->userData().setLiterial("rotate", std::move(rotate));
        prim->userData().setLiterial("shape", std::move(shape));

        set_output("prim", std::move(prim));
    }
};

ZENO_DEFNODE(LightNode)({
    {
        {"vec3f", "position", "0, 0, 0"},
        {"vec3f", "scale", "1, 1, 1"},
        {"vec3f", "rotate", "0, 0, 0"},
        {"vec3f", "color", "1, 1, 1"},
        {"float", "intensity", "1"},
        {"int", "islight", "1"},
        {"int", "invertdir", "1"}
    },
    {
        "prim"
    },
    {
        {"enum Disk Plane", "Shape", "Plane"},
    },
    {"shader"},
});
}