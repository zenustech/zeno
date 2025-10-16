#include <zeno/zeno.h>
#include <zeno/types/GeometryObject.h>
#include <zeno/geo/geometryutil.h>
#include "glm/gtc/matrix_transform.hpp"
#include <zeno/formula/zfxexecute.h>
#include <zeno/core/FunctionManager.h>
#include <sstream>
#include <zeno/formula/zfxexecute.h>
#include <zeno/utils/format.h>
#include <unordered_set>
#include <zeno/para/parallel_reduce.h>
#include <zeno/utils/interfaceutil.h>


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace zeno {
    using namespace zeno::reflect;

    struct CopyToPoints : INode {
        void apply() override {
            auto _input_object = ZImpl(get_input2<GeometryObject_Adapter>("Input Geometry"));
            auto _target_Obj = ZImpl(get_input2<GeometryObject_Adapter>("Target Geometry"));
            std::string alignTo = ZImpl(get_input2<std::string>("Align To"));

            if (!_input_object) {
                throw makeError<UnimplError>("empty input object.");
            }
            else if (!_target_Obj) {
                throw makeError<UnimplError>("empty target object.");
            }

            auto input_object = _input_object->m_impl.get();
            auto target_Obj = _target_Obj->m_impl.get();
            if (!input_object->has_point_attr("pos")) {
                throw makeError<UnimplError>("invalid input object.");
            }
            else if (!target_Obj->has_point_attr("pos")) {
                throw makeError<UnimplError>("invalid target object.");
            }

            std::vector<zeno::vec3f> inputPos = input_object->get_attrs<zeno::vec3f>(ATTR_POINT, "pos");
            std::vector <std::tuple< bool, std::vector<int >> > inputFacesPoints(input_object->nfaces(), std::tuple<bool, std::vector<int >>({ false, std::vector<int>() }));
            for (int i = 0; i < input_object->nfaces(); ++i) {
                inputFacesPoints[i] = std::tuple(input_object->isLineFace(i), input_object->face_points(i));
            }

            std::vector<zeno::vec3f> inputNrm;
            std::vector<zeno::vec2f> inputLines;
            bool hasNrm = input_object->has_point_attr("nrm");
            bool isLine = input_object->is_Line();
            if (hasNrm) {
                inputNrm = input_object->get_attrs<zeno::vec3f>(ATTR_POINT, "nrm");
            }

            zeno::vec3f originCenter, _min, _max;
            std::tie(_min, _max) = geomBoundingBox(input_object);
            if (alignTo == "Align To Point Center") {
                originCenter = (_min + _max) / 2;
            }
            else {
                originCenter = _min;
            }

            size_t targetObjPointsCount = target_Obj->npoints();
            size_t inputObjPointsCount = input_object->npoints();
            size_t inputObjFacesCount = input_object->nfaces();
            size_t newObjPointsCount = targetObjPointsCount * inputObjPointsCount;
            size_t newObjFacesCount = targetObjPointsCount * inputObjFacesCount;

            std::vector<zeno::vec3f> newObjPos(newObjPointsCount, zeno::vec3f());
            std::vector<zeno::vec3f> newObjNrm;
            if (hasNrm) {
                newObjNrm.resize(newObjPointsCount);
            }
            std::vector<std::vector<int>> faces;
            std::vector<size_t> pts_offset, faces_offset;

            for (size_t i = 0; i < targetObjPointsCount; ++i) {
                zeno::vec3f targetPos = target_Obj->get_elem<zeno::vec3f>(ATTR_POINT, "pos", 0, i);
                zeno::vec3f dx = targetPos - originCenter;
                glm::mat4 translate = glm::translate(glm::mat4(1.0), glm::vec3(dx[0], dx[1], dx[2]));

                const size_t pt_offset = i * inputObjPointsCount;
                const size_t face_offset = i * inputObjFacesCount;
                for (size_t j = 0; j < inputObjPointsCount; j++)
                {
                    auto idx = pt_offset + j;

                    auto& pt = inputPos[j];
                    glm::vec4 gp = translate * glm::vec4(pt[0], pt[1], pt[2], 1);
                    newObjPos[idx] = zeno::vec3f(gp.x, gp.y, gp.z);

                    if (hasNrm) {
                        newObjNrm[idx] = inputNrm[j];
                    }
                }
                for (size_t j = 0; j < inputObjFacesCount; ++j)
                {
                    std::vector<int> facePoints = std::get<1>(inputFacesPoints[j]);
                    for (int k = 0; k < facePoints.size(); ++k) {
                        facePoints[k] += pt_offset;
                    }
                    //TODO: line如何考虑？
                    faces.push_back(facePoints);
                    //spgeo->set_face(face_offset + j, stdVec2zeVec(facePoints), !std::get<0>(inputFacesPoints[j]));
                }
                pts_offset.push_back(pt_offset);
                faces_offset.push_back(face_offset);
            }

            auto spgeo = create_GeometryObject(Topo_IndiceMesh, input_object->is_base_triangle(), newObjPos, faces);
            if (hasNrm) {
                spgeo->create_attr(ATTR_POINT, "nrm", newObjNrm);
            }
            for (int i = 0; i < pts_offset.size(); i++) {
                spgeo->inheritAttributes(_input_object.get(), -1, pts_offset[i], {"pos", "nrm"}, faces_offset[i], {});
            }
            ZImpl(set_output("Output", std::move(spgeo)));
        }
    };

    ZENDEFNODE(CopyToPoints, 
    {
        {
            {gParamType_Geometry, "Input Geometry"},
            {gParamType_Geometry, "Target Geometry"},
            ParamPrimitive("Align To", gParamType_String, "Align To Point Center", Combobox, std::vector<std::string>{"Align To Point Center", "Align To Min"}),
        },
        {
            {gParamType_Geometry, "Output"},
        },
        {},
        {"deprecated"},
    });


    struct Sweep : INode {

#if 0
        const float w = surface_width;
        if (surface_shape == "Square Tube") {
            if (input_object->is_Line()) {
                //先针对单位长度的竖线，以及给定的width和column,建模出基本的tube模型
                std::vector<vec3f> basepts;
                for (int iy = 0; iy <= 0/*只须考虑底部的面*/; iy++) {
                    for (int iz = 0; iz <= surface_column; iz++) {
                        for (int ix = 0; ix <= surface_column; ix++) {
                            if (0 < ix && ix < surface_column &&
                                0 < iz && iz < surface_column) {
                                continue;
                            }
                            float step = surface_width / surface_column;
                            float start = -surface_width / 2;
                            basepts.push_back(vec3f(start + ix * step, iy, start + iz * step));
                        }
                    }
                }

                const int npts = basepts.size();
                std::vector<vec3f> pts = input_object->points_pos();
                std::vector<vec3f> lastendface(npts);
                vec3f originVec(0, 1, 0);

                std::vector<vec3f> newpts;
                for (int currPt = 0; currPt < pts.size(); currPt++) {
                    int nextPt = input_object->getLineNextPt(currPt);
                    if (currPt != nextPt) {
                        //能进到这里，就不是最后一个点
                        vec3f p1 = pts[currPt];
                        vec3f p2 = pts[nextPt];

                        float vx = p2[0] - p1[0];
                        float vy = p2[1] - p1[1];
                        float vz = p2[2] - p1[2];
                        float LB = zeno::sqrt(vx * vx + vy * vy + vz * vz);

                        vec3f vpt(vx, vy, vz);
                        vpt = zeno::normalize(vpt);
#if 1
                        glm::vec3 vxz(vpt[0], 0, vpt[2]);

                        //步骤一：绕x轴，将y轴旋转至z轴原来位置
                        glm::mat4 M_toz(1.0);
                        {
                            M_toz = glm::rotate(M_toz, glm::pi<float>() / 2, glm::vec3(1, 0, 0));
                        }

                        //步骤二：旋转至xz平面的投影
                        glm::mat4 M_toxz(1.0);
                        {
                            float theta = zeno::atan(vxz[0] / vxz[2]);
                            glm::vec3 rotate_u = glm::vec3(0, 1, 0);
                            M_toxz = glm::rotate(M_toxz, theta, rotate_u);   //绕y轴旋转theta度，将向量投影至x轴
                        }

                        //步骤三：往上提
                        glm::mat4 M_up(1.0);
                        if (vpt[1] != 0)
                        {
                            glm::vec3 vdir(vpt[0], vpt[1], vpt[2]);
                            float tantheta = vpt[1] / zeno::sqrt(vpt[0] * vpt[0] + vpt[2] * vpt[2]);
                            float theta = zeno::atan(tantheta);
                            glm::vec3 rotate_u = glm::cross(glm::vec3(vpt[0], 0, vpt[2]), vdir);
                            M_up = glm::rotate(M_up, theta, rotate_u);
                        }

                        //步骤四：平移
                        glm::mat4 M_translate(1.0);
                        M_translate = glm::translate(M_translate, glm::vec3(p1[0], p1[1], p1[2]));

                        glm::mat4 transM = M_translate * M_up * M_toxz * M_toz;
#endif

#if 0
                        //步骤一：先平移至原点
                        glm::mat4 M_translate(1.0);
                        M_translate = glm::translate(M_translate, -glm::vec3(p1[0], p1[1], p1[2]));

                        //步骤二：投影至xz平面
                        glm::mat4 M_toxz(1.0);
                        {
                            glm::vec3 vdir(vpt[0], vpt[1], vpt[2]);
                            float tantheta = vdir[1] / zeno::sqrt(vdir[0] * vdir[0] + vdir[2] * vdir[2]);
                            float theta = zeno::atan(tantheta);

                            glm::vec3 rotate_u = glm::cross(vdir, glm::vec3(vpt[0], 0, vpt[2]));
                            M_toxz = glm::rotate(M_toxz, theta, rotate_u);
                        }

                        //步骤三：投影至z轴
                        glm::mat4 M_z(1.0);
                        {
                            glm::vec3 vxz(vpt[0], 0, vpt[2]);
                            float theta = zeno::atan(vxz[0] / vxz[2]);
                            glm::vec3 rotate_u = glm::vec3(0, 1, 0);
                            M_z = glm::rotate(M_z, theta, rotate_u);   //绕y轴逆向旋转theta度，将向量投影至x轴
                        }

                        //步骤四：旋转至y轴
                        glm::mat4 M_y(1.0);
                        {
                            M_y = glm::rotate(M_y, -glm::pi<float>() / 2, glm::vec3(1, 0, 0));
                        }
                        glm::mat4 rev_transM = M_y * M_z * M_toxz * M_translate;
                        glm::mat4 transM = glm::inverse(rev_transM);
#endif


#if 0
                        //计算两个向量的叉积，得到旋转轴
                        vec3f rot = zeno::cross(originVec, vpt);
                        auto len = zeno::length(rot);
                        glm::mat4 Rx(1.0);
                        if (len == 0) {
                            //考虑originVec和vpt平行的情况
                            //只要单位矩阵即可
                        }
                        else {
                            rot = zeno::normalize(rot);

                            float cos_theta = zeno::dot(originVec, vpt);
                            float sin_theta = zeno::sqrt(1 - cos_theta * cos_theta);
                            float rx = rot[0], ry = rot[1], rz = rot[2];

                            glm::mat3 K = glm::mat3(
                                0, rz, -ry,
                                -rz, 0, rx,
                                ry, -rx, 0
                            );
                            glm::mat3 R_ = glm::mat3(1.0) + sin_theta * K + (1 - cos_theta) * K * K;
                            Rx = glm::mat4(R_);
                        }

                        glm::mat4 transM(
                            Rx[0][0], Rx[0][1], Rx[0][2], 0,
                            Rx[1][0], Rx[1][1], Rx[1][2], 0,
                            Rx[2][0], Rx[2][1], Rx[2][2], 0,
                            p1[0], p1[1], p1[2], 1);
#endif

                        std::vector<vec3f> istartface;
                        {
                            istartface.resize(npts);
                            for (int i = 0; i < npts; i++) {
                                vec3f basept = basepts[i];
                                glm::vec4 v(basept[0], basept[1], basept[2], 1.0);

#if 1
                                //M_translate* M_up* M_toxz* M_toz;
                                glm::vec4 _v1 = M_toz * v;
                                glm::vec4 _v2 = M_toxz * _v1;
                                glm::vec4 _v3 = M_up * _v2;
                                glm::vec4 _v4 = M_translate * _v3;
#endif

                                glm::vec4 newpt = transM * v;
                                istartface[i] = vec3f(newpt[0], newpt[1], newpt[2]);
                            }
                        }
                        if (currPt >= 0) {
                            if (false) {
                                //此时收集到istartface是当前点，需要考虑到上一线段的末尾点形成面与当前面的不一致情况
                                int _npoints = newpts.size();
                                assert(_npoints >= npts);
                                for (int i = 0; i < npts; i++) {
                                    //修正当前面
                                    istartface[i] = (istartface[i] + lastendface[i]) / 2;
                                }
                            }
                            //istartface = lastendface;
                        }

                        //先收集这些确定的点
                        if (currPt >= 0) {
                            for (auto pt : istartface) {
                                newpts.push_back(pt);
                            }
                        }

                        std::vector<vec3f> iendface(npts);
                        for (int i = 0; i < npts; i++) {
                            iendface[i] = istartface[i] + LB * vpt;
                            lastendface[i] = iendface[i];
                        }

                        //提前收集endface
                        for (auto pt : iendface) {
                            newpts.push_back(pt);
                        }
                    }
                    else {
                        //for (auto pt : lastendface) {
                        //    newpts.push_back(pt);
                        //}
                    }
                }
                std::shared_ptr<GeometryObject> spgeo = std::make_unique<GeometryObject>(false, newpts.size(), 0);
                spgeo->create_attr(ATTR_POINT, "pos", newpts);
                return spgeo;
#if 0
                std::vector<vec3f> pts = input_object->points_pos();
                for (int i = 0; i < pts.size(); i++) {
                    if (i > 0 && i < pts.size() - 1) {
                        vec3f pt = pts[i];
                        vec3f pt_1 = pts[i - 1];
                        vec3f pt1 = pts[i + 1];
                        vec3f v1 = zeno::normalize(pt1 - pt);
                        vec3f v2 = zeno::normalize(pt_1 - pt);
                        while (zeno::dot(v1, v2) == -1) {
                            v1 += vec3f(0.0001, 0, 0);
                            v2 += vec3f(0.0001, 0, 0);
                        }

                        vec3f hor_v = zeno::normalize(v1 + v2);
                        vec3f ver_v = zeno::normalize(zeno::cross(v1, hor_v));

                        //这个start_pt留到最后加
                        vec3f start_pt = pt + w / 2.f * (hor_v + ver_v);
                        //逆时针转一圈
                        vec3f curr_start = start_pt;
                        vec3f curr_pt;
                        for (int i = 1; i <= surface_column; i++) {
                            curr_pt = curr_start + w / surface_column * i * (-hor_v);
                            int j;
                            j = 0;
                        }
                        curr_start = curr_pt; //上一次遍历的终点是下一个线段的起点
                        for (int i = 1; i <= surface_column; i++) {
                            curr_pt = curr_start + w / surface_column * i * (-ver_v);
                            int j;
                            j = 0;
                        }
                        curr_start = curr_pt;
                        for (int i = 1; i <= surface_column; i++) {
                            curr_pt = curr_start + w / surface_column * i * hor_v;
                            int j;
                            j = 0;
                        }
                        curr_start = curr_pt;
                        for (int i = 1; i <= surface_column; i++) {
                            curr_pt = curr_start + w / surface_column * i * ver_v;
                            int j;
                            j = 0;
                        }
                    }
                }
#endif
            }
        }
        return input_object;
#endif
        void apply() override {
            //TODO
        }
    };

    ZENDEFNODE(Sweep,
    {
        {
            ParamObject("Input", gParamType_Geometry),
            ParamPrimitive("Snap Distance", gParamType_Float, 0.2f),
            ParamPrimitive("Surface Shape", gParamType_String, "Square Tube", Combobox, std::vector<std::string>{"Second Input", "Square Tube"}),
            ParamPrimitive("Width", gParamType_Float, 0.2, Slider, std::vector<float>{0.0, 1.0, 0.001}, "visible = parameter('Surface Shape').value == 'Square Tube';"),
            ParamPrimitive("Columns", gParamType_Int, 2, Slider, std::vector<int>{1, 20, 1}, "visible = parameter('Surface Shape').value == 'Square Tube';")
        },
        {
            {gParamType_Geometry, "Output"},
        },
        {},
        {"deprecated"},
    });


    struct Merge : INode {
        void apply() override
        {
            auto list_object = ZImpl(get_input2<zeno::ListObject>("Input Of Objects"));
            auto mergedObj = zeno::mergeObjects(list_object.get());
            ZImpl(set_output("Output", std::move(mergedObj)));
        }
    };

    ZENDEFNODE(Merge,
    {
        {
            ParamObject("Input Of Objects", gParamType_List)
        },
        {
            {gParamType_Geometry, "Output"},
        },
        {},
        {"deprecated"},
    });


    struct Divide : INode {
        void apply() override {
            auto input_object = ZImpl(get_input2<GeometryObject_Adapter>("Input"));
            zeno::vec3f Size = ZImpl(get_input2<zeno::vec3f>("Size"));
            bool remove_shared_edge = ZImpl(get_input2<bool>("Remove Shared Edges"));

            auto bbox = geomBoundingBox(input_object->m_impl.get());
            float xmin = bbox.first[0], ymin = bbox.first[1], zmin = bbox.first[2],
                xmax = bbox.second[0], ymax = bbox.second[1], zmax = bbox.second[2];
            float dx = Size[0], dy = Size[1], dz = Size[2];
            int nx = (dx == 0) ? 0 : (xmax - xmin) / dx;
            int ny = (dy == 0) ? 0 : (ymax - ymin) / dy;
            int nz = (dz == 0) ? 0 : (zmax - zmin) / dz;

            zany output_obj;
            for (int i = 0; i <= nx; i++) {
                float xi = xmin + i * dx;
                output_obj = divideObject(input_object.get(), Keep_Both, vec3f(xi, 0, 0), vec3f(1, 0, 0));
            }
            for (int i = 0; i <= ny; i++) {
                float yi = ymin + i * dy;
                output_obj = divideObject(input_object.get(), Keep_Both, vec3f(0, yi, 0), vec3f(0, 1, 0));
            }
            for (int i = 0; i <= nz; i++) {
                float zi = zmin + i * dz;
                output_obj = divideObject(input_object.get(), Keep_Both, vec3f(0, 0, zi), vec3f(0, 0, 1));
            }
            ZImpl(set_output("Output", std::move(output_obj)));
        }
    };
    ZENDEFNODE(Divide,
    {
        {
            ParamObject("Input", gParamType_Geometry),
            ParamPrimitive("Size", gParamType_Vec3f, zeno::vec3f(0,0,0)),
            ParamPrimitive("Remove Shared Edges", gParamType_Bool, false)
        },
        {
            {gParamType_Geometry, "Output"},
        },
        {},
        {"deprecated"},
    });


    struct Resample : INode {
        void apply() override
        {
            auto input_object = ZImpl(get_input2<GeometryObject_Adapter>("Input Object"));
            float Length = ZImpl(get_input2<float>("Length"));

            if (Length == 0) {
                ZImpl(set_output("Output", std::move(input_object)));
                return;
            }

            const int nface = input_object->m_impl->nfaces();
            const auto& pos = input_object->m_impl->points_pos();

            std::vector<vec3f> newpos;
            std::vector<std::vector<int>> newfaces;
            for (int iFace = 0; iFace < nface; iFace++) {
                std::vector<int> face_indice = input_object->m_impl->face_points(iFace);
                std::vector<int> newface;

                for (int i = 0; i < face_indice.size(); i++) {
                    int startPt = -1, endPt = -1;
                    if (i == 0) {
                        startPt = face_indice[face_indice.size() - 1];
                        endPt = face_indice[0];
                    }
                    else {
                        startPt = face_indice[i - 1];
                        endPt = face_indice[i];
                    }

                    vec3f startPos = pos[startPt], endPos = pos[endPt];
                    vec3f dir = zeno::normalize(endPos - startPos);
                    int k = zeno::length(endPos - startPos) / Length;

                    for (int j = 0; j < k; j++) {
                        vec3f middlePos = startPos + j * Length * dir;
                        newface.push_back(newpos.size());
                        newpos.push_back(middlePos);
                        //if (j == k) {
                        //    if (!(middlePos == endPos)) {
                        //        newface.push_back(newpos.size());
                        //        newpos.push_back(endPos);
                        //    }
                        //}
                    }
                }
                newfaces.push_back(newface);
            }

            auto spOutput = create_GeometryObject(Topo_IndiceMesh, input_object->is_base_triangle(), newpos, newfaces);
            ZImpl(set_output("Output", std::move(spOutput)));
        }
    };
    ZENDEFNODE(Resample,
    {
        {
            ParamObject("Input Object", gParamType_Geometry),
            ParamPrimitive("Length", gParamType_Float, 0.1f)
        },
        {
            {gParamType_Geometry, "Output"},
        },
        {},
        {"deprecated"},
    });



    struct  Reverse : INode {
        void apply() override
        {
            auto input_object = ZImpl(get_input2<GeometryObject_Adapter>("Input Object"));
            const int nface = input_object->nfaces();
            const auto& pos = input_object->points_pos();
            std::vector<std::vector<int>> faces;
            for (int iFace = 0; iFace < nface; iFace++) {
                std::vector<int> face_indice = input_object->m_impl->face_points(iFace);
                std::reverse(face_indice.begin() + 1, face_indice.end());
                faces.emplace_back(std::move(face_indice));
            }
            auto spOutput = create_GeometryObject(Topo_IndiceMesh, input_object->is_base_triangle(), pos, faces);
            ZImpl(set_output("Output", std::move(spOutput)));
        }
    };
    ZENDEFNODE(Reverse,
    {
        {
            ParamObject("Input Object", gParamType_Geometry)
        },
        {
            {gParamType_Geometry, "Output"},
        },
        {},
        {"deprecated"},
    });


    struct Clip : INode {
        void apply() override
        {
            auto input_object = ZImpl(get_input2<GeometryObject_Adapter>("Input Object"));
            std::string Keep = ZImpl(get_input2<std::string>("Keep"));
            zeno::vec3f center_pos = ZImpl(get_input2<zeno::vec3f>("Center Position"));
            zeno::vec3f direction = ZImpl(get_input2<zeno::vec3f>("Direction"));

            DivideKeep keep;
            if (Keep == "All") {
                keep = Keep_Both;
            }
            else if (Keep == "Part Below The Plane") {
                keep = Keep_Above;
            }
            else if (Keep == "Part Above The Plane") {
                keep = Keep_Below;
            }
            else {
                throw makeError<UnimplError>("Keep Error");
            }
            auto spOutput = divideObject(input_object.get(), keep, center_pos, direction);
            ZImpl(set_output("Output", std::move(spOutput)));
        }
    };
    ZENDEFNODE(Clip,
    {
        {
            ParamObject("Input Object", gParamType_Geometry),
            ParamPrimitive("Keep", gParamType_String, "All", Combobox, std::vector<std::string>{"All", "Part Below The Plane", "Part Above The Plane"}),
            ParamPrimitive("Center Position", gParamType_Vec3f, zeno::vec3f(0, 0, 0)),
            ParamPrimitive("Direction", gParamType_Vec3f, zeno::vec3f(0, 1, 0))
        },
        {
            {gParamType_Geometry, "Output"},
        },
        {},
        {"deprecated"},
    });



    struct RemoveUnusedPoints : INode {
        void apply() override {
            auto input_object = ZImpl(get_input<GeometryObject_Adapter>("Input Object"));
            std::set<int> unused;
            for (int i = 0; i < input_object->npoints(); i++) {
                if (input_object->m_impl->point_faces(i).empty()) {
                    unused.insert(i);
                }
            }
            for (auto iter = unused.rbegin(); iter != unused.rend(); iter++) {
                //TODO: impl remove_points
                input_object->remove_point(*iter);
            }
            ZImpl(set_output("Output", std::move(input_object)));
        }
    };
    ZENDEFNODE(RemoveUnusedPoints,
    {
        {
            ParamObject("Input Object", gParamType_Geometry)
        },
        {
            {gParamType_Geometry, "Output"},
        },
        {},
        {"geom"},
    });


    struct RemoveInlinePoints : INode {

        void apply() override
        {
            auto input_object = ZImpl(get_input<GeometryObject_Adapter>("Input Object"));

            const std::vector<vec3f> pos = input_object->m_impl->points_pos();
            std::set<int> unusedPoints;
            for (int iFace = 0; iFace < input_object->nfaces(); iFace++) {
                const std::vector<int>& pts = input_object->m_impl->face_points(iFace);
                for (int i = 1; i < pts.size() - 1; i++) {
                    //观察pts[i]是否只有一个面
                    const auto& _faces = input_object->point_faces(pts[i]);
                    if (_faces.size() != 1) {
                        continue;
                    }
                    assert(_faces[0] == iFace);

                    vec3f pa = pos[pts[i - 1]];
                    vec3f pb = pos[pts[i]];
                    vec3f pc = pos[pts[i + 1]];
                    //检查pa,pb,pc是否共线
                    if (zeno::dot(zeno::normalize(pb - pa), zeno::normalize(pc - pa)) == 1) {
                        unusedPoints.insert(pts[i]);
                    }
                }
            }
            for (auto iter = unusedPoints.rbegin(); iter != unusedPoints.rend(); iter++) {
                //TODO: impl remove_points
                input_object->remove_point(*iter);
            }

            ZImpl(set_output("Output", std::move(input_object)));
        }
    };
    ZENDEFNODE(RemoveInlinePoints,
    {
        {
            ParamObject("Input Object", gParamType_Geometry)
        },
        {
            {gParamType_Geometry, "Output"},
        },
        {},
        {"geom"},
    });



    struct Measure : INode {

        void apply() override
        {
            auto input_object = ZImpl(get_input2<GeometryObject_Adapter>("Input Object"));
            std::string measure = ZImpl(get_input2<std::string>("Measure"));
            std::string outputAttrName = ZImpl(get_input2<std::string>("Output Attribute Name"));

            const auto& pos = input_object->m_impl->points_pos();
            int nFace = input_object->m_impl->nfaces();
            std::vector<float> measurements(nFace, 0.f);

            for (int iFace = 0; iFace < nFace; iFace++) {
                const std::vector<int>& pts = input_object->m_impl->face_points(iFace);
                int n = pts.size();
                std::vector<vec3f> face_ptpos(n);
                for (int i = 0; i < n; i++) {
                    face_ptpos[i] = pos[pts[i]];
                }
                if (measure == "Area") {
                    if (n < 3) continue;        //线段
                    vec3f P0 = face_ptpos[0];
                    float area = 0.f;
                    for (int i = 1; i < pts.size() - 1; i++) {
                        vec3f P1 = face_ptpos[i];
                        vec3f P2 = face_ptpos[i + 1];

                        vec3f v1 = P1 - P0;
                        vec3f v2 = P2 - P0;

                        vec3f cross_product = zeno::cross(v1, v2);
                        area += 0.5f * zeno::length(cross_product);
                    }
                    measurements[iFace] = area;
                }
                else if (measure == "Length") {
                    //TODO
                }
            }
            input_object->create_face_attr(stdString2zs(outputAttrName), measurements);
            ZImpl(set_output("Output", std::move(input_object)));
        }
    };
    ZENDEFNODE(Measure,
    {
        {
            ParamObject("Input Object", gParamType_Geometry),
            ParamPrimitive("Measure", gParamType_String, "Area", Combobox, std::vector<std::string>{"Area", "Length"}),
            ParamPrimitive("Output Attribute Name", gParamType_String, "area")
        },
        {
            {gParamType_Geometry, "Output"},
        },
        {},
        {"geom"},
    });


    struct Mirror : INode {
        void apply() override
        {
            auto input_object = ZImpl(get_input2<GeometryObject_Adapter>("Input Object"));
            bool bKeepOriginal = ZImpl(get_input2<bool>("Keep Original"));
            zeno::vec3f Position = ZImpl(get_input2<zeno::vec3f>("Position"));
            zeno::vec3f Direction = ZImpl(get_input2<zeno::vec3f>("Direction"));
            float Distance = ZImpl(get_input2<float>("Distance"));

            const auto& pos = input_object->m_impl->points_pos();
            int npos = pos.size();
            int nface = input_object->nfaces();

            int Npos = bKeepOriginal ? 2 * npos : npos;
            int Nface = bKeepOriginal ? 2 * nface : nface;

            std::vector<vec3f> new_pos(Npos);
            std::vector<std::vector<int>> faces;
            faces.reserve(nface);

            std::copy(pos.begin(), pos.end(), new_pos.begin());
            if (bKeepOriginal) {
                std::copy(pos.begin(), pos.end(), new_pos.begin() + npos);
            }

            if (bKeepOriginal) {
                for (int iFace = 0; iFace < nface; iFace++) {
                    std::vector<int> pts = input_object->m_impl->face_points(iFace);
                    faces.emplace_back(pts);
                }
            }

            for (int iFace = 0; iFace < nface; iFace++) {
                std::vector<int> pts = input_object->m_impl->face_points(iFace);
                std::reverse(pts.begin(), pts.end());
                std::vector<int> offset_pts;
                for (int pt : pts) {
                    zeno::vec3f P = pos[pt];

                    float A = Direction[0], B = Direction[1], C = Direction[2];
                    float px = Position[0], py = Position[1], pz = Position[2];
                    float D = -(A * px + B * py + C * pz);
                    zeno::vec3f normal_vec(A, B, C);
                    float normal_squared = zeno::dot(normal_vec, normal_vec);
                    float dist = (zeno::dot(P, normal_vec) + D) / normal_squared;
                    zeno::vec3f P_mirror = P - 2 * dist * normal_vec;

                    pt += (bKeepOriginal ? npos : 0);
                    new_pos[pt] = P_mirror;
                    offset_pts.push_back(pt);
                }
                faces.emplace_back(offset_pts);
            }

            auto spOutput = create_GeometryObject(Topo_IndiceMesh, input_object->is_base_triangle(), new_pos, faces);
            ZImpl(set_output("Output", std::move(input_object)));
        }
    };
    ZENDEFNODE(Mirror,
    {
        {
            ParamObject("Input Object", gParamType_Geometry),
            ParamPrimitive("Position", gParamType_Vec3f, zeno::vec3f(0, 0, 0)),
            ParamPrimitive("Direction", gParamType_Vec3f, zeno::vec3f(1, 0, 0)),
            ParamPrimitive("Distance", gParamType_Float, 1.f),
            ParamPrimitive("Keep Original", gParamType_Bool, true)
        },
        {
            {gParamType_Geometry, "Output"},
        },
        {},
        {"geom"},
    });


    struct MatchSize : INode {

        static glm::vec3 mapplypos(glm::mat4 const& matrix, zeno::vec3f const& vector) {
            auto vec = zeno::vec_to_other<glm::vec3>(vector);
            auto vector4 = matrix * glm::vec4(vec, 1.0f);
            return glm::vec3(vector4) / vector4.w;
        }

        static glm::vec3 mapplynrm(glm::mat4 const& matrix, zeno::vec3f const& vector) {
            auto vec = zeno::vec_to_other<glm::vec3>(vector);
            glm::mat3 normMatrix(matrix);
            normMatrix = glm::transpose(glm::inverse(normMatrix));
            auto vector3 = normMatrix * vec;
            return glm::normalize(vector3);
        }

        void apply() override
        {
            auto input_object = ZImpl(get_input<GeometryObject_Adapter>("Input Object"));
            auto match_object = ZImpl(get_input<GeometryObject_Adapter>("Match Object"));
            zeno::vec3f TargetPosition = ZImpl(get_input2<zeno::vec3f>("Target Position"));
            zeno::vec3f TargetSize = ZImpl(get_input2<zeno::vec3f>("Target Size"));
            bool bTranslate = ZImpl(get_input2<bool>("Translate"));
            std::string TranslateXTo = ZImpl(get_input2<std::string>("Translate X To"));
            std::string TranslateYTo = ZImpl(get_input2<std::string>("Translate Y To"));
            std::string TranslateZTo = ZImpl(get_input2<std::string>("Translate Z To"));
            bool bScaleToFit = ZImpl(get_input2<bool>("Scale To Fit"));

            if (!bTranslate && !bScaleToFit) {
                ZImpl(set_output("Output", std::move(input_object)));
                return;
            }

            const auto& currbbox = geomBoundingBox(input_object->m_impl.get());

            zeno::vec3f boxmin, boxmax;
            if (match_object) {
                const auto& bbox = geomBoundingBox(match_object->m_impl.get());
                boxmin = bbox.first;
                boxmax = bbox.second;
            }
            else {
                boxmin = TargetPosition - 0.5 * TargetSize;
                boxmax = TargetPosition + 0.5 * TargetSize;
            }

            vec3f boxsize = boxmax - boxmin;
            if (TranslateXTo == "Min") {
                boxmin[0] += boxsize[0] / 2.;
                boxmax[0] += boxsize[0] / 2.;
            }
            else if (TranslateXTo == "Max") {
                boxmin[0] -= boxsize[0] / 2.;
                boxmax[0] -= boxsize[0] / 2.;
            }

            if (TranslateYTo == "Min") {
                boxmin[1] += boxsize[1] / 2.;
                boxmax[1] += boxsize[1] / 2.;
            }
            else if (TranslateYTo == "Max") {
                boxmin[1] -= boxsize[1] / 2.;
                boxmax[1] -= boxsize[1] / 2.;
            }

            if (TranslateZTo == "Min") {
                boxmin[2] += boxsize[2] / 2.;
                boxmax[2] += boxsize[2] / 2.;
            }
            else if (TranslateZTo == "Max") {
                boxmin[2] -= boxsize[2] / 2.;
                boxmax[2] -= boxsize[2] / 2.;
            }

            glm::mat4 matrix(1.);
            if (bScaleToFit) {
                vec3f boxsize = (boxmax - boxmin);
                matrix = glm::scale(matrix, glm::vec3(boxsize[0], boxsize[1], boxsize[2]));
            }
            if (bTranslate) {
                vec3f boxcenter = (boxmin + boxmax) / 2.;
                vec3f center_offset = boxcenter - (currbbox.first + currbbox.second) / 2.;
                matrix = glm::translate(matrix, glm::vec3(center_offset[0], center_offset[1], center_offset[2]));
            }

            if (input_object->has_attr(ATTR_POINT, "pos"))
            {
                input_object->m_impl->foreach_attr_update<zeno::vec3f>(ATTR_POINT, "pos", 0, [&](int idx, zeno::vec3f old_pos)->zeno::vec3f {
                    auto p = mapplypos(matrix, old_pos);
                    auto newpos = zeno::other_to_vec<3>(p);
                    return newpos;
                    });
            }

            if (input_object->has_attr(ATTR_POINT, "nrm"))
            {
                input_object->m_impl->foreach_attr_update<zeno::vec3f>(ATTR_POINT, "nrm", 0, [&](int idx, zeno::vec3f old_nrm)->zeno::vec3f {
                    auto n = mapplynrm(matrix, old_nrm);
                    auto newnrm = zeno::other_to_vec<3>(n);
                    return newnrm;
                    });
            }
            ZImpl(set_output("Output", std::move(input_object)));
        }
    };
    ZENDEFNODE(MatchSize,
    {
        {
            ParamObject("Input Object", gParamType_Geometry),
            ParamObject("Match Object", gParamType_Geometry),
            ParamPrimitive("Target Position", gParamType_Vec3f, zeno::vec3f(0,0,0), Vec3edit, Any(), "visible = parameter('Match Object').connected == false;"),
            ParamPrimitive("Target Size", gParamType_Vec3f, zeno::vec3f(1,1,1), Vec3edit, Any(), "visible = parameter('Match Object').connected == false;"),
            ParamPrimitive("Translate", gParamType_Bool),
            ParamPrimitive("Translate X To", gParamType_String, "Center", Combobox, std::vector<std::string>{"Min", "Center", "Max"}),
            ParamPrimitive("Translate Y To", gParamType_String, "Center", Combobox, std::vector<std::string>{"Min", "Center", "Max"}),
            ParamPrimitive("Translate Z To", gParamType_String, "Center", Combobox, std::vector<std::string>{"Min", "Center", "Max"}),
            ParamPrimitive("Scale To Fit", gParamType_Bool)
        },
        {
            {gParamType_Geometry, "Output"},
        },
        {},
        {"geom"},
    });


    struct Peak : INode {

        void apply() override
        {
            auto input_object = ZImpl(get_input2<GeometryObject_Adapter>("Input Object"));
            float Distance = ZImpl(get_input2<float>("Distance"));

            const std::vector<vec3f>& nrms = input_object->m_impl->get_attrs<zeno::vec3f>(ATTR_POINT, "nrm");
            input_object->m_impl->foreach_attr_update<zeno::vec3f>(ATTR_POINT, "pos", 0, [&](int idx, zeno::vec3f old_pos)->zeno::vec3f {
                const vec3f& nrm = nrms[idx];
                if (idx == 3) {
                    int j;
                    j = 0;
                }
                return old_pos + nrm * Distance;
            });

            ZImpl(set_output("Output", std::move(input_object)));
        }
    };
    ZENDEFNODE(Peak,
    {
        {
            ParamObject("Input Object", gParamType_Geometry),
            {gParamType_Float, "Distance", "0.2"}
        },
        {
            {gParamType_Geometry, "Output"},
        },
        {},
        {"geom"},
    });


    struct Scatter : INode {
        void apply() override
        {
            auto input_object = ZImpl(get_input2<GeometryObject_Adapter>("Input"));
            int Count = ZImpl(get_input2<int>("Points Count"));
            int Seed = ZImpl(get_input2<int>("Random Seed"));
            std::string sampleRegion = ZImpl(get_input2<std::string>("Sample Regin"));

            auto spOutput = zeno::scatter(input_object.get(), sampleRegion, Count, Seed);
            ZImpl(set_output("Output", std::move(spOutput)));
        }
    };
    ZENDEFNODE(Scatter,
    {
        {
            ParamObject("Input", gParamType_Geometry),
            ParamPrimitive("Points Count", gParamType_Int, 10, Slider, std::vector<int>{0, 1000, 1}),
            ParamPrimitive("Random Seed", gParamType_Int, 0, Slider, std::vector<int>{0, 100, 1}),
            ParamPrimitive("Sample Regin",gParamType_String, "Face", Combobox, std::vector<std::string>{"Face", "Volumn"})
        },
        {
            {gParamType_Geometry, "Output"},
        },
        {},
        {"geom"},
    });


    struct ConvertLine : INode {
        void apply() override
        {
            //TODO:
#if 0
            auto input_object = ZImpl(get_input2<GeometryObject_Adapter>("Input Object"));
            bool bKeepOrder = ZImpl(get_input2<bool>("Keep Order"));
            std::string lengthAttr = ZImpl(get_input2<std::string>("Length Attribute"));

            int nPts = input_object->npoints();
            const std::vector<std::vector<int>>& faces = input_object->m_impl->face_indice();

            std::unordered_set<uint64_t> hash;
            std::vector<std::vector<int>> lines;
            for (auto ind : faces) {
                for (int i = 1; i < ind.size(); i++) {
                    int from = ind[i - 1];
                    int to = ind[i];
                    std::pair<int, int> p = { from, to };
                    uint64_t minidx = std::min(from, to), maxidx = std::max(from, to);
                    uint64_t id = minidx + (maxidx << 32);
                    if (hash.find(id) == hash.end()) {
                        lines.push_back({ from, to });
                        hash.insert(id);
                    }
                }
            }

            auto line_object = create_GeometryObject(false, nPts, lines.size());
            line_object->create_point_attr("pos", input_object->points_pos());
            for (auto line : lines) {
                line_object->m_impl->add_face(line, false);
            }

            ZImpl(set_output("Output", std::move(line_object)));
#endif
        }
    };
    ZENDEFNODE(ConvertLine,
    {
        {
            ParamObject("Input Object", gParamType_Geometry),
            ParamPrimitive("Keep Order", gParamType_Bool),
            ParamPrimitive("Length Attribute", gParamType_String)
        },
        {
            {gParamType_Geometry, "Output"},
        },
        {},
        {"geom"},
    });


    struct UniquePoints : INode {
        void apply() override
        {
            auto input_object = ZImpl(get_input2<GeometryObject_Adapter>("Input"));
            bool bPostComputeNormals = ZImpl(get_input2<bool>("Post-Compute Normals"));

            std::vector<std::vector<vec3f>> newFaces;
            const std::vector<zeno::vec3f>& pos = input_object->m_impl->points_pos();
            std::vector<vec3f> point_normals;

            for (int iFace = 0; iFace < input_object->nfaces(); iFace++) {
                std::vector<int> pts = input_object->m_impl->face_points(iFace);
                std::vector<vec3f> new_face(pts.size());
                for (int i = 0; i < pts.size(); i++) {
                    new_face[i] = pos[pts[i]];
                }
                newFaces.push_back(new_face);

                if (bPostComputeNormals && new_face.size() > 2) {
                    vec3f p01 = new_face[1] - new_face[0];
                    vec3f p12 = new_face[2] - new_face[1];
                    vec3f nrm = zeno::normalize(zeno::cross(p01, p12));
                    for (int i = 0; i < new_face.size(); i++) {
                        point_normals.push_back(nrm);
                    }
                }
            }
            auto spOutput = constructGeom(newFaces);
            if (bPostComputeNormals)
                spOutput->create_point_attr("nrm", point_normals);

            ZImpl(set_output("Output", std::move(spOutput)));
        }
    };
    ZENDEFNODE(UniquePoints,
    {
        {
            ParamObject("Input", gParamType_Geometry),
            ParamPrimitive("Post-Compute Normals", gParamType_Bool)
        },
        {
            {gParamType_Geometry, "Output"},
        },
        {},
        {"geom"},
    });


    struct  Sort : INode {
        void apply() override
        {
            auto input_object = ZImpl(get_input2<GeometryObject_Adapter>("Input"));
            std::string pointSort = ZImpl(get_input2<std::string>("Point Sort"));
            zeno::vec3f alongVector = ZImpl(get_input2<vec3f>("Vector"));
            zeno::vec3f nearPoint = ZImpl(get_input2<vec3f>("Point"));
            bool bReversePointSort = ZImpl(get_input2<bool>("Reverse Point Sort"));
            std::string faceSort = ZImpl(get_input2<std::string>("Face Sort"));
            std::string byAttrName = ZImpl(get_input2<std::string>("Attribute Name"));
            bool bReverseFaceSort = ZImpl(get_input2<bool>("Reverse Face Sort"));

            int npts = input_object->m_impl->npoints();
            int nfaces = input_object->m_impl->nfaces();
            const auto& pos = input_object->m_impl->points_pos();

            if (pointSort == "By Vertex Order") {
                std::set<std::string> edges;
                std::vector<vec3f> new_pos(pos.size());
                std::vector<std::vector<int>> faces;
                faces.reserve(nfaces);
                int nSortPoints = 0;
                std::map<int, int> old2new;
                
                for (int iFace = 0; iFace < nfaces; iFace++) {
                    std::vector<int> pts = input_object->m_impl->face_points(iFace);
                    std::vector<int> newface;
                    for (int vertex = 0; vertex < pts.size(); vertex++) {
                        int fromPt, toPt;

                        fromPt = pts[vertex];

                        if (vertex == pts.size() - 1) {
                            toPt = pts[0];
                        }
                        else {
                            toPt = pts[vertex + 1];
                        }

                        int newFrom, newTo;
                        if (old2new.find(fromPt) != old2new.end()) {
                            newFrom = old2new[fromPt];
                        }
                        else {
                            newFrom = nSortPoints++;
                        }
                        if (old2new.find(toPt) != old2new.end()) {
                            newTo = old2new[toPt];
                        }
                        else {
                            newTo = nSortPoints++;
                        }

                        old2new.insert(std::make_pair(fromPt, newFrom));
                        old2new.insert(std::make_pair(toPt, newTo));
                        newface.push_back(newFrom);
                    }
                    faces.emplace_back(newface);
                }
                for (int oldpt = 0; oldpt < pos.size(); oldpt++) {
                    int newpt = old2new[oldpt];
                    new_pos[newpt] = pos[oldpt];
                }

                auto spOutput = create_GeometryObject(Topo_IndiceMesh, false, new_pos, faces);
                ZImpl(set_output("Output", std::move(spOutput)));
            }
        }
    };
    ZENDEFNODE(Sort,
    {
        {
            ParamObject("Input", gParamType_Geometry),
            ParamPrimitive("Point Sort", gParamType_String, "No Change", Combobox, std::vector<std::string>{"No Change", "By Vertex Order", "Random", "Along Vector", "Near To Point"}),
            ParamPrimitive("Vector", gParamType_Vec3f, zeno::vec3f(0,1,0), Vec3edit, Any(), "visible = parameter('Point Sort').value == 'Along Vector';"),
            ParamPrimitive("Point",  gParamType_Vec3f, zeno::vec3f(0,1,0), Vec3edit, Any(), "visible = parameter('Point Sort').value == 'Near To Point';"),
            ParamPrimitive("Reverse Point Sort", gParamType_Bool),
            ParamPrimitive("Face Sort", gParamType_String, "No Change", Combobox, std::vector<std::string>{"No Change", "Random", "By Attribute"}),
            ParamPrimitive("Attribute Name", gParamType_String, "", Lineedit, Any(), "visible = parameter('Face Sort').value == 'By Attribute';"),
            ParamPrimitive("Reverse Face Sort", gParamType_Bool),
        },
        {
            {gParamType_Geometry, "Output"},
        },
        {},
        {"geom"},
    });


    struct PolyExpand : INode {
        void findOffsetPoints(
            vec3f p1,
            vec3f p2,
            vec3f p3,
            float offset,
            vec3f& p1_inside,
            vec3f& p1_outside,
            vec3f& p2_inside,
            vec3f& p2_outside)
        {
            vec3f v12 = p2 - p1, v23 = p3 - p2;
            vec3f n1 = zeno::cross(v12, v23);
            vec3f n2 = zeno::cross(v23, v12);
            vec3f p1_inside_dir = zeno::normalize(zeno::cross(n1, v12));
            vec3f p1_outside_dir = -p1_inside_dir;
            vec3f p2_outside_dir = zeno::normalize(zeno::cross(n2, v23));
            vec3f p2_inside_dir = -p2_outside_dir;
            p1_inside = p1 + p1_inside_dir * offset;
            p1_outside = p1 + p1_outside_dir * offset;
            p2_inside = p2 + p2_inside_dir * offset;
            p2_outside = p2 + p2_outside_dir * offset;
        }

        vec3f findIntersect(vec3f p1, vec3f v1, vec3f p2, vec3f v2)
        {
            float x1 = p1[0], y1 = p1[1], z1 = p1[2], x2 = p2[0], y2 = p2[1], z2 = p2[2];
            float a1 = v1[0], b1 = v1[1], c1 = v1[2], a2 = v2[0], b2 = v2[1], c2 = v2[2];

            float v1_dist = zeno::lengthSquared(v1);
            float v2_dist = zeno::lengthSquared(v2);
            float v1_v2 = zeno::dot(v1, v2);

            float divider = v1_dist * v2_dist - zeno::pow(v1_v2, 2);

            float t = (zeno::dot(p2-p1,v1) * v2_dist - zeno::dot(p2-p1,v2) * v1_v2) / divider;
            float s = (zeno::dot(p2 - p1, v2) * v1_dist - zeno::dot(p2 - p1, v1) * v1_v2) / divider;

            vec3f result = p1 + t * v1;
            return result;
        }

        void apply() override
        {
            auto input_object = ZImpl(get_input<GeometryObject_Adapter>("Input"));
            float Offset = ZImpl(get_input2<float>("Offset"));
            bool bOutputInside = ZImpl(get_input2<bool>("Output Inside"));
            bool bOutputOutside = ZImpl(get_input2<bool>("Output Outside"));

            const std::vector<vec3f>& pos = input_object->m_impl->points_pos();
            std::vector<std::vector<vec3f>> newFaces;
            for (int iFace = 0; iFace < input_object->nfaces(); iFace++) {
                std::vector<int> pts = input_object->m_impl->face_points(iFace);
                std::vector<vec3f> inside_face(pts.size()), outside_face(pts.size());
                for (int i = 0; i < pts.size(); i++) {
                    int p1 = pts[i];
                    int p2 = pts[(i + 1) % pts.size()];
                    int p3 = pts[(i + 2) % pts.size()];

                    vec3f p1_inside, p1_outside, p2_inside, p2_outside;

                    findOffsetPoints(pos[p1], pos[p2], pos[p3], Offset,
                        p1_inside, p1_outside, p2_inside, p2_outside);
                    vec3f v1 = zeno::normalize(pos[p2] - pos[p1]);
                    vec3f v2 = zeno::normalize(pos[p3] - pos[p2]);
                    vec3f pi_inside = findIntersect(p1_inside, v1, p2_inside, v2);
                    vec3f pi_outside = findIntersect(p1_outside, v1, p2_outside, v2);
                    inside_face[i] = pi_inside;
                    outside_face[i] = pi_outside;
                }
                if (bOutputInside)
                    newFaces.push_back(inside_face);
                if (bOutputOutside)
                    newFaces.push_back(outside_face);
            }
            auto spOutput = constructGeom(newFaces);
            auto spFinal = zeno::fuseGeometry(spOutput.get(), 0.005);
            ZImpl(set_output("Output", std::move(spFinal)));
        }
    };
    ZENDEFNODE(PolyExpand,
    {
        {
            ParamObject("Input", gParamType_Geometry),
            ParamPrimitive("Output Inside", gParamType_Bool, false),
            ParamPrimitive("Output Outside", gParamType_Bool, false)
        },
        {
            {gParamType_Geometry, "Output"},
        },
        {},
        {"geom"},
    });


    struct Extrude : INode {

        zeno::vec3f getSpreadDirector(GeometryObject* input_obj, int idxPt)
        {
            /* 根据给定的输入obj，以及顶点，得到这个点的挤出延长线方向向量 */
            //找出这个点关联的所有面，然后得到各个面的法线，然后求和即可
            std::vector<int> faces = input_obj->point_faces(idxPt);
            zeno::vec3f nrm_sum(0,0,0);
            for (auto f : faces) {
                //TODO: 后续可以考虑提前缓存法线，目前基于实现方便先直接调用
                zeno::vec3f face_nrm = input_obj->face_normal(f);
                nrm_sum += face_nrm;
            }
            return zeno::normalize(nrm_sum);
        }

        std::pair<zeno::vec3f, zeno::vec3f> getExtrudeDest(
            GeometryObject* input_obj,
            int lastPt,
            int currPt,
            int back_face,
            float dist,
            bool& needFace
            )
        {
            /* 给定基准的back平面，以及这个平面上相邻的两个点lastPt,currPt，以dist的挤出距离，求出:
            1. 这两个点延长至front平面时的终点位置
            2. 这两个点的线在延申时形成的面，是否需要产生。（如果是两个面的共线，则不需要产生）
             */
            zeno::vec3f nrm = input_obj->face_normal(back_face);
            const auto& pts = input_obj->points_pos();

            zeno::vec3f last_pos = pts[lastPt];
            zeno::vec3f last_spread_dir = getSpreadDirector(input_obj, lastPt);
            float costheta = zeno::dot(last_spread_dir, nrm);
            float k = 1. / costheta;
            zeno::vec3f dest1 = k * dist * last_spread_dir + last_pos;

            zeno::vec3f curr_pos = pts[currPt];
            zeno::vec3f curr_spread_dir = getSpreadDirector(input_obj, currPt);
            costheta = zeno::dot(curr_spread_dir, nrm);
            k = 1. / costheta;
            zeno::vec3f dest2 = k * dist * curr_spread_dir + curr_pos;

            std::vector<int> faces1 = input_obj->point_faces(lastPt);
            std::vector<int> faces2 = input_obj->point_faces(currPt);
            //两个点关联的面，除了back_face，观察是否还有别的面。
            needFace = true;
            for (auto f1 : faces1) {
                for (auto f2 : faces2) {
                    if (f1 == f2 && f1 != back_face) {
                        needFace = false;
                        break;
                    }
                }
            }
            return { dest1, dest2 };
        }

        void apply() override {
            auto input_object = ZImpl(get_input<GeometryObject_Adapter>("Input"));
            float Distance = ZImpl(get_input2<float>("Distance"));
            float Inset = ZImpl(get_input2<float>("Inset"));
            bool bOutputFrontAttr = ZImpl(get_input2<bool>("Output Front Attribute"));
            std::string front_attr = ZImpl(get_input2<std::string>("Front Attribute"));
            bool bOutputBackAttr = ZImpl(get_input2<bool>("Output Back Attribute"));
            std::string back_attr = ZImpl(get_input2<std::string>("Back Attribute"));

            if (Distance > 0) {
                std::vector<std::vector<vec3f>> newFaces;
                const std::vector<zeno::vec3f>& pos = input_object->m_impl->points_pos();
                for (int iFace = 0; iFace < input_object->m_impl->nfaces(); iFace++) {
                    std::vector<int> pts = input_object->m_impl->face_points(iFace);
                    std::vector<vec3f> dest_pts;

                    std::vector<vec3f> front_pts;
                    for (int i = 0; i < pts.size(); i++) {
                        int lastPt = -1, currPt = -1;
                        if (i == 0) {
                            lastPt = pts[pts.size() - 1];
                            currPt = pts[0];
                        }
                        else {
                            lastPt = pts[i - 1];
                            currPt = pts[i];
                        }
                        bool needFace = false;
                        const auto& spread = getExtrudeDest(input_object->m_impl.get(), lastPt, currPt, iFace, Distance, needFace);
                        zeno::vec3f last_spread = spread.first;
                        zeno::vec3f curr_spread = spread.second;

                        front_pts.push_back(last_spread);

                        if (needFace) {
                            //dest_pts[i], dest_pts[i-1], pts[i], pts[i-1]将构成一个平面
                            std::vector<zeno::vec3f> new_face = { curr_spread, last_spread, pos[lastPt], pos[currPt] };
                            newFaces.push_back(new_face);
                        }
                    }
                    //front face
                    newFaces.push_back(front_pts);  //todo: 需要传一个front的字段标记为front面。
                }
                auto spOutput = constructGeom(newFaces);
                auto spFinal = zeno::fuseGeometry(spOutput.get(), 0.005);
                ZImpl(set_output("Output", std::move(spFinal)));
            }
            else {
                ZImpl(set_output("Output", std::move(input_object)));
            }
        }
    };
    ZENDEFNODE(Extrude,
    {
        {
            ParamObject("Input", gParamType_Geometry),
            ParamPrimitive("Distance", gParamType_Float, 0.f, Slider, std::vector<float>{0.0, 1.0, 0.01}),
            ParamPrimitive("Inset", gParamType_Float, 0.f, Slider, std::vector<float>{0.0, 1.0, 0.01}),
            ParamPrimitive("Output Front Attribute", gParamType_Bool, false, Checkbox),
            ParamPrimitive("Front Attribute", gParamType_String, "extrudeFront", Lineedit, zeno::reflect::Any(), "enable = parameter('Output Front Attribute').value == 1;"),
            ParamPrimitive("Output Back Attribute", gParamType_Bool, false, Checkbox),
            ParamPrimitive("Back Attribute", gParamType_String, "extrudeBack", Lineedit, zeno::reflect::Any(), "enable = parameter('Output Back Attribute').value == 1;")
        },
        {
            {gParamType_Geometry, "Output"},
        },
        {},
        {"geom"},
    });


    struct Blast : INode {

        void apply() override
        {
            auto input_object = ZImpl(get_input2<GeometryObject_Adapter>("Input"));
            std::string zfx = ZImpl(get_input2<std::string>("Zfx Expression"));
            std::string group = ZImpl(get_input2<std::string>("Group"));
            bool deleteNonSelected = ZImpl(get_input2<bool>("Delete Non Selected"));

            zeno::ZfxContext ctx;
            std::string rem_what;
            std::string idnum;
            std::string nor = deleteNonSelected ? "!" : "";
            if (group == "Points") {
                rem_what = "point";
                idnum = "@ptnum";
                ctx.runover = ATTR_POINT;
            }
            else if (group == "Faces") {
                rem_what = "face";
                idnum = "@facenum";
                ctx.runover = ATTR_FACE;
            }
            else {
                throw UnimplError("Unknown group on blast");
            }

            std::string finalZfx;
            finalZfx += "if (" + nor + "(" + zfx + ")" + ") {\n";
            finalZfx += "    remove_" + rem_what + "(" + idnum + ");\n";
            finalZfx += "}";

            auto spOutput = input_object->clone();

            ctx.spNode = m_pAdapter;
            ctx.spObject = std::move(spOutput);
            ctx.code = finalZfx;

            zeno::ZfxExecute zfxexe(finalZfx, &ctx);
            zfxexe.execute();

            ZImpl(set_output("Output", std::move(ctx.spObject)));
        }

        zeno::CustomUI export_customui() const override {
            zeno::CustomUI ui = zeno::INode::export_customui();
            ui.uistyle.background = "#DF7C1B";
            return ui;
        }
    };
    ZENDEFNODE(Blast,
    {
        {
            ParamObject("Input", gParamType_Geometry),
            ParamPrimitive("Zfx Expression", gParamType_String),
            ParamPrimitive("Group", gParamType_String, "Points", Combobox, std::vector<std::string>{"Points", "Faces"}),
            ParamPrimitive("Delete Non Selected", gParamType_Bool)
        },
        {
            {gParamType_Geometry, "Output"},
        },
        {},
        {"geom"},
    });



    struct AverageFuse : INode {
        //houdini fuse节点的average模式

        void apply() override
        {
            auto input_object = ZImpl(get_input2<GeometryObject_Adapter>("Input Geometry"));
            float snapDistance = ZImpl(get_input2<float>("Snap Distance"));
            if (!input_object) {
                throw makeError<UnimplError>("empty input object.");
            }
            auto spOutput = fuseGeometry(input_object.get(), snapDistance);
            ZImpl(set_output("Output", std::move(spOutput)));
        }
    };

    ZENDEFNODE(AverageFuse,
    {
        {
            ParamObject("Input Geometry", gParamType_Geometry),
            ParamPrimitive("Snap Distance", gParamType_Float, 0.01f)
        },
        {
            {gParamType_Geometry, "Output"},
        },
        {},
        {"geom"},
    });

    struct LakeHouseSettings : zeno::INode
    {
        void apply() override
        {

        }
    };
    ZENDEFNODE(LakeHouseSettings,
    {
        {
            ParamPrimitive("Global Seed", gParamType_Int, 2983, Slider, std::vector<int>{1, 4000, 1}),
            {gParamType_Float, "Roof Window Prob", "0.6"},
            {gParamType_Float, "Chimney Prob", "0.1"},
            {gParamType_Float, "Tower Prob", "0.6"}
        },
        {
        },
        {},
        {"geom"},
    });

}
