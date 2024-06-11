#include <zeno/zeno.h>
#include <zeno/extra/TempNode.h>
#include <zeno/utils/eulerangle.h>

#include <zeno/types/UserData.h>
#include <zeno/types/LightObject.h>
#include <zeno/types/PrimitiveObject.h>

#include <glm/glm.hpp>
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtx/quaternion.hpp>
#include "glm/gtc/matrix_transform.hpp"

namespace zeno {

struct LightNode : INode {
    virtual void apply() override {
        auto isL = true; //get_input2<int>("islight");
        auto invertdir = get_input2<int>("invertdir");
        
        auto scale = get_input2<zeno::vec3f>("scale");
        auto rotate = get_input2<zeno::vec3f>("rotate");
        auto position = get_input2<zeno::vec3f>("position");
        auto quaternion = get_input2<zeno::vec4f>("quaternion");

        auto color = get_input2<zeno::vec3f>("color");
        auto exposure = get_input2<float>("exposure");
        auto intensity = get_input2<float>("intensity");

        auto scaler = powf(2.0f, exposure);

        if (std::isnan(scaler) || std::isinf(scaler) || scaler < 0.0f) {
            scaler = 1.0f;
            printf("Light exposure = %f is invalid, fallback to 0.0 \n", exposure);
        }

        intensity *= scaler;

        auto ccc = color * intensity;
        for (size_t i=0; i<ccc.size(); ++i) {
            if (std::isnan(ccc[i]) || std::isinf(ccc[i]) || ccc[i] < 0.0f) {
                ccc[i] = 1.0f;
                printf("Light color component %lu is invalid, fallback to 1.0 \n", i);
            }
        }

        auto mask = get_input2<int>("mask");
        auto spread = get_input2<zeno::vec2f>("spread");
        auto visible = get_input2<int>("visible");
        auto doubleside = get_input2<int>("doubleside");

        std::string type = get_input2<std::string>(lightTypeKey);
        auto typeEnum = magic_enum::enum_cast<LightType>(type).value_or(LightType::Diffuse);
        auto typeOrder = magic_enum::enum_integer(typeEnum);

        std::string shapeString = get_input2<std::string>(lightShapeKey);
        auto shapeEnum = magic_enum::enum_cast<LightShape>(shapeString).value_or(LightShape::Plane);
        auto shapeOrder = magic_enum::enum_integer(shapeEnum);

        auto prim = std::make_shared<zeno::PrimitiveObject>();
        auto &VERTS = prim->verts;
        auto &LINES = prim->lines;
        auto &TRIS = prim->tris;

        if (has_input("prim")) {
            auto mesh = get_input<PrimitiveObject>("prim");

            if (mesh->tris->size() > 0) {
                prim = mesh;
                shapeEnum = LightShape::TriangleMesh;
                shapeOrder = magic_enum::enum_integer(shapeEnum);
            }
        } else {

            auto order = get_input2<std::string>("EulerRotationOrder:");
            auto orderTyped = magic_enum::enum_cast<EulerAngle::RotationOrder>(order).value_or(EulerAngle::RotationOrder::YXZ);

            auto measure = get_input2<std::string>("EulerAngleMeasure:");
            auto measureTyped = magic_enum::enum_cast<EulerAngle::Measure>(measure).value_or(EulerAngle::Measure::Radians);

            glm::vec3 eularAngleXYZ = glm::vec3(rotate[0], rotate[1], rotate[2]);
            glm::mat4 rotation = EulerAngle::rotate(orderTyped, measureTyped, eularAngleXYZ);

            if (shapeEnum == LightShape::Point) {
                scale = {0 ,scale[1], 0};

                if (typeEnum == LightType::Diffuse) {
                    spread = {1, 1};
                }
            }

            const auto transformWithoutScale = [&]() { 
                glm::quat rawQuat(quaternion[0], quaternion[1], quaternion[2], quaternion[3]);
                glm::mat4 matQuat  = glm::toMat4(rawQuat);

                glm::mat4 transform = glm::translate(glm::mat4(1.0f), glm::vec3(position[0], position[1], position[2]));
                transform = transform * rotation * matQuat;
                return transform;
            } ();

            VERTS->push_back(zeno::vec3f(+0.5, 0, +0.5));
            VERTS->push_back(zeno::vec3f(+0.5, 0, -0.5));
            VERTS->push_back(zeno::vec3f(-0.5, 0, +0.5));
            VERTS->push_back(zeno::vec3f(-0.5, 0, -0.5));

            auto pscale = std::max(scale[0], scale[2]);
            pscale = std::max(pscale, scale[1]);

            if (shapeEnum == LightShape::Sphere) {

                auto tmpPrim = zeno::TempNodeSimpleCaller("CreateSphere")
                        .set2<zeno::vec3f>("position", {0,0,0})
                        .set2<zeno::vec3f>("scaleSize", {1,1,1})
                        .set2<float>("radius", 0.5f)
                        .set2<zeno::vec3f>("rotate", {0,0,0})
                        .set2<bool>("hasNormal", false)
                        .set2<bool>("hasVertUV", false)
                        .set2<bool>("isFlipFace", false)
                        .set2<int>("rows", 180)
                        .set2<int>("columns", 360)
                        .set2<bool>("quads", false)
                        .set2<bool>("SphereRT", false)
                        .set2<std::string>("EulerRotationOrder:", "XYZ")
                        .set2<std::string>("EulerAngleMeasure:", "Degree")
                        .call().get<zeno::PrimitiveObject>("prim");

                        VERTS->reserve(tmpPrim->verts->size());
                        TRIS.reserve(tmpPrim->tris->size());

                VERTS->insert(VERTS.end(), tmpPrim->verts->begin(), tmpPrim->verts->end());
                for (size_t i=0; i<tmpPrim->tris.size(); ++i) {
                    auto tri = tmpPrim->tris[i];
                    TRIS.push_back(tri+4);
                }

                scale = zeno::vec3f(min(scale[0], scale[2]));
                pscale = 0.0;
            } 
            else if (shapeEnum == LightShape::Ellipse) {

                auto tmpPrim = zeno::TempNodeSimpleCaller("CreateDisk")
                        .set2<zeno::vec3f>("position", {0,0,0})
                        .set2<zeno::vec3f>("scaleSize", {1,1,1})
                        .set2<zeno::vec3f>("rotate", {0,0,0})
                        .set2<float>("radius", 0.5f)
                        .set2<float>("divisions", 360)
                        .set2<bool>("hasNormal", false)
                        .set2<bool>("hasVertUV", false)
                        .set2<bool>("isFlipFace", false)
                        .call().get<zeno::PrimitiveObject>("prim");

                        VERTS->reserve(tmpPrim->verts->size());
                        TRIS.reserve(tmpPrim->tris->size());

                VERTS->insert(VERTS.end(), tmpPrim->verts->begin(), tmpPrim->verts->end());
                for (size_t i=0; i<tmpPrim->tris.size(); ++i) {
                    auto tri = tmpPrim->tris[i];
                    TRIS.push_back(tri+4);
                }
            }

            if (shapeEnum != LightShape::Point) {
                // Plane Indices
                if (TRIS->size() == 0) {
                    TRIS.emplace_back(zeno::vec3i(0, 3, 1));
                    TRIS.emplace_back(zeno::vec3i(3, 0, 2));
                }

                for (auto& v : VERTS) {
                    v = scale * v;
                }
            }

            auto line_spread = spread;
            if (typeEnum != LightType::Projector) {
                line_spread = {spread[0], spread[0]};
            } 

            int lut[] = {+1, +3, -1, -3};

            int vertex_offset = VERTS->size();            
            for (size_t i=0; i<4; ++i) {

                auto info = lut[i];

                auto axis = glm::vec3(0, 0, 0);
                auto pick = 2-(abs(info)-1);
                axis[pick] = std::copysign(1, info);

                if (pick == 0) { // inverse axis
                    axis[pick] *= -1;
                }

                glm::mat4 sub_rotate = glm::rotate(glm::mat4(1.0), line_spread[i%2] * M_PIf/2.0f, axis);
                auto end_point = sub_rotate * glm::vec4(0, -0.3, 0, 1.0f);

                glm::vec4 p0 = glm::vec4(0,0,0,1);
                glm::vec4 p1 = glm::vec4(pscale, pscale, pscale, 1.0f) * (end_point);

                auto delta = glm::vec4(0.0);
                delta[abs(info)-1] = 0.5f * scale[abs(info)-1];

                if ( std::signbit(info) ) { // negative
                    delta = -delta;
                }

                p0 += delta;
                p1 += delta;

                if (line_spread[i%2] < line_spread[(i+1)%2]) { // spread at the same surface
                    p1 *= cos( line_spread[(i+1)%2] * M_PIf/2.0f ) / cos( line_spread[(i)%2] * M_PIf/2.0f );
                }

                VERTS->push_back(zeno::vec3f(p0[0], p0[1], p0[2]));
                VERTS->push_back(zeno::vec3f(p1[0], p1[1], p1[2]));

                LINES->push_back({vertex_offset, vertex_offset+1});
                vertex_offset +=2;
            }

            if (shapeEnum == LightShape::Point) {

                int anchor_offset = VERTS->size();
                VERTS->push_back({0,0,0});

                if (typeEnum != LightType::Projector){

                    glm::mat4 sub_trans = glm::rotate(glm::mat4(1.0), M_PIf/4, glm::vec3(0,1,0));

                    for (size_t i=4; i<=(anchor_offset-1); ++i) {
                        auto& v = VERTS.at(i);
                        auto p = sub_trans * glm::vec4(v[0], v[1], v[2], 1);

                        VERTS->push_back( {p.x, p.y, p.z} );
                        LINES->push_back({anchor_offset, (int)VERTS.size()-1});
                    }
                } else {
                    auto vertical_distance = VERTS[anchor_offset-1][1];
                    float x_max=-FLT_MAX, x_min=FLT_MAX;
                    float z_max=-FLT_MAX, z_min=FLT_MAX;

                    for (int i=0; i<4; ++i) {
                        auto idx = anchor_offset - 1 - i * 2;
                        auto& tmp = VERTS[idx];

                        x_max = max(tmp[0], x_max);
                        x_min = min(tmp[0], x_min);    
                        z_max = max(tmp[2], z_max);
                        z_min = min(tmp[2], z_min);
                    }

                    VERTS->push_back({ x_max, vertical_distance, z_max} );
                    VERTS->push_back({ x_max, vertical_distance, z_min} );
                    VERTS->push_back({ x_min, vertical_distance, z_min} );
                    VERTS->push_back({ x_min, vertical_distance, z_max} );

                    LINES->push_back({anchor_offset+1, anchor_offset+2});
                    LINES->push_back({anchor_offset+2, anchor_offset+3});
                    LINES->push_back({anchor_offset+3, anchor_offset+4});
                    LINES->push_back({anchor_offset+4, anchor_offset+1});
                }

                if (typeEnum == LightType::Diffuse) {
                
                    int vertex_offset = VERTS->size();
                    
                    for (auto i : {-1, 0, 1}) {
                        for (auto j : {-1, 0, 1}) {

                            auto sub_trans = glm::rotate(glm::mat4(1.0), M_PIf/4, glm::vec3(i,0,j));
                            if (i == 0 && j == 0) { sub_trans = glm::mat4(1.0); }

                            sub_trans = glm::scale(sub_trans, {0, scale[1], 0});

                            auto p1 = sub_trans * glm::vec4(0, +.3, 0, 1);
                            auto p2 = sub_trans * glm::vec4(0, -.3, 0, 1);  

                            VERTS->push_back(zeno::vec3f(p1[0], p1[1], p1[2]));
                            VERTS->push_back(zeno::vec3f(p2[0], p2[1], p2[2]));

                            LINES->push_back({anchor_offset, vertex_offset+0});
                            LINES->push_back({anchor_offset, vertex_offset+1});

                            vertex_offset += 2;
                        } // j
                    } // i 
                }   
            }

            if ( (shapeEnum != LightShape::Sphere) && (invertdir || doubleside) ) {

                auto sub_trans = glm::rotate(glm::mat4(1.0), M_PIf, glm::vec3(1,0,0));
                auto vertices_offset = VERTS.size();

                if (doubleside) {

                    LINES->reserve(LINES->size()*2);
                    VERTS.reserve(VERTS->size()*2);
                    typeof(LINES) tmp(LINES->size());

                    std::transform(LINES.begin(), LINES.end(), tmp.begin(), 
                    [&](auto ele){ return ele + vertices_offset; });

                    LINES->insert(LINES.end(), tmp->begin(), tmp->end());
                }

                for (size_t i=0; i<vertices_offset; ++i) {
                    auto& v = VERTS.at(i);
                    auto p = sub_trans * glm::vec4(v[0], v[1], v[2], 1.0f);
                    if (invertdir) {
                        v = zeno::vec3f(p[0], p[1], p[2]);
                    }
                    if (doubleside) {
                        VERTS->push_back(zeno::vec3f(p[0], p[1], p[2]));
                    }
                }
            }

            auto &clr = VERTS.add_attr<zeno::vec3f>("clr");
            for (size_t i=0; i<VERTS.size(); ++i) {
                auto& v = VERTS.at(i);
                auto p = transformWithoutScale * glm::vec4(v[0], v[1], v[2], 1.0f);
                v = zeno::vec3f(p[0], p[1], p[2]);
                clr[i] = ccc;
            } 
        }

        auto& ud = prim->userData();

        ud.set2("isRealTimeObject", std::move(isL));

        ud.set2("isL", std::move(isL));
        ud.set2("ivD", std::move(invertdir));
        ud.set2("pos", std::move(position));
        ud.set2("scale", std::move(scale));
        ud.set2("rotate", std::move(rotate));
        ud.set2("quaternion", std::move(quaternion));
        ud.set2("color", std::move(color));
        ud.set2("intensity", std::move(intensity));

        auto fluxFixed = get_input2<float>("fluxFixed");
        ud.set2("fluxFixed", std::move(fluxFixed));
        auto maxDistance = get_input2<float>("maxDistance");
        ud.set2("maxDistance", std::move(maxDistance));
        auto falloffExponent = get_input2<float>("falloffExponent");
        ud.set2("falloffExponent", std::move(falloffExponent));

        if (has_input2<std::string>("profile")) {
            auto profile = get_input2<std::string>("profile");
            ud.set2("lightProfile", std::move(profile));
        }
        if (has_input2<std::string>("texturePath")) {
            auto texture = get_input2<std::string>("texturePath");
            ud.set2("lightTexture", std::move(texture));

            auto gamma = get_input2<float>("textureGamma");
            ud.set2("lightGamma", std::move(gamma));
        }

        ud.set2("type", std::move(typeOrder));
        ud.set2("shape", std::move(shapeOrder));

        ud.set2("mask", std::move(mask));
        ud.set2("spread", std::move(spread));
        ud.set2("visible", std::move(visible));
        ud.set2("doubleside", std::move(doubleside));

        auto visibleIntensity = get_input2<float>("visibleIntensity");
        ud.set2("visibleIntensity", std::move(visibleIntensity));

        set_output("prim", std::move(prim));
    }

    const static inline std::string lightShapeKey = "shape";

    static std::string lightShapeDefaultString() {
        auto name = magic_enum::enum_name(LightShape::Plane);
        return std::string(name);
    }

    static std::string lightShapeListString() {
        auto list = magic_enum::enum_names<LightShape>();

        std::string result;
        for (auto& ele : list) {
            result += " ";
            result += ele;
        }
        return result;
    }

    const static inline std::string lightTypeKey = "type";

    static std::string lightTypeDefaultString() {
        auto name = magic_enum::enum_name(LightType::Diffuse);
        return std::string(name);
    }

    static std::string lightTypeListString() {
        auto list = magic_enum::enum_names<LightType>();

        std::string result;
        for (auto& ele : list) {
            result += " ";
            result += ele;
        }
        return result;
    }
};

ZENO_DEFNODE(LightNode)({
    {
        {"vec3f", "position", "0, 0, 0"},
        {"vec3f", "scale", "1, 1, 1"},
        {"vec3f", "rotate", "0, 0, 0"},
        {"vec4f", "quaternion", "1, 0, 0, 0"},

        {"colorvec3f", "color", "1, 1, 1"},
        {"float", "exposure", "0"},
        {"float", "intensity", "1"},
        {"float", "fluxFixed", "-1.0"},

        {"vec2f", "spread", "1.0, 0.0"},
        {"float", "maxDistance", "-1.0" },
        {"float", "falloffExponent", "2.0"},
        {"int", "mask", "255"},
        {"bool", "visible", "0"},
        {"bool", "invertdir", "0"},
        {"bool", "doubleside", "0"},

        {"readpath", "profile"},
        {"readpath", "texturePath"},
        {"float",  "textureGamma", "1.0"},
        {"float", "visibleIntensity", "-1.0"},

        {"enum " + LightNode::lightShapeListString(), LightNode::lightShapeKey, LightNode::lightShapeDefaultString()},
        {"enum " + LightNode::lightTypeListString(), LightNode::lightTypeKey, LightNode::lightTypeDefaultString()},
        {"PrimitiveObject", "prim"},
    },
    {
        "prim"
    },
    {
        {"enum " + EulerAngle::RotationOrderListString(), "EulerRotationOrder", EulerAngle::RotationOrderDefaultString()},
        {"enum " + EulerAngle::MeasureListString(), "EulerAngleMeasure", EulerAngle::MeasureDefaultString()}
    },
    {"shader"},
});

} // namespace