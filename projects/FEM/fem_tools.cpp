#include "declares.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/quaternion.hpp>

#include "../ZenoFX/LinearBvh.h"

namespace zeno {

struct MatrixObject : zeno::IObject{
    std::variant<glm::mat3, glm::mat4> m;
};
// compact all the CV's items into one single primitive object
// mainly serves for matrix-free solver, and no storage for connectivity matrix is needed
struct MakeFEMPrimitive : zeno::INode {
    virtual void apply() override {
        // input prim
        auto prim = get_input<zeno::PrimitiveObject>("prim");
        auto YoungModulus = get_input2<float>("Stiffness");
        auto PossonRatio = get_input2<float>("VolumePreserve");
        // the strength of position driven mechanics
        auto ExamShapeCoeff = get_input2<float>("ExamShapeCoeff");
        // the strength of barycentric interpolator-driven mechanics
        // the interpolator can be driving bones or skin etc.
        auto EmbedShapeCoeff = get_input2<float>("EmbedShapeCoeff");
        
        // Add nodal-wise channel
        const auto& pos = prim->add_attr<zeno::vec3f>("pos");
        auto& curPos = prim->verts.add_attr<zeno::vec3f>("curPos");
        auto& curVel = prim->verts.add_attr<zeno::vec3f>("curVel");
        auto& prePos = prim->verts.add_attr<zeno::vec3f>("prePos");
        auto& preVel = prim->verts.add_attr<zeno::vec3f>("preVel");

        std::copy(pos.begin(),pos.end(),curPos.begin());
        std::copy(pos.begin(),pos.end(),prePos.begin());
        std::fill(curVel.begin(),curVel.end(),zeno::vec3f(0,0,0));
        std::fill(preVel.begin(),preVel.end(),zeno::vec3f(0,0,0));

        auto& examW = prim->verts.add_attr<float>("examW",ExamShapeCoeff);
        auto& examShape = prim->verts.add_attr<zeno::vec3f>("examShape");
        std::copy(pos.begin(),pos.end(),examShape.begin());

        // Add element-wise channel
        auto& density = prim->quads.add_attr<float>("phi",1000);
        auto& E = prim->quads.add_attr<float>("E",YoungModulus);
        auto& nu = prim->quads.add_attr<float>("nu",PossonRatio);
        auto& v = prim->quads.add_attr<float>("v",0);
        auto& use_dynamic = prim->quads.add_attr<float>("dynamic_mark",0);

        // characteristic norm for scalability 
        auto& cnorm = prim->quads.add_attr<float>("cnorm",0);
        // element-wise vol
        auto& vol = prim->quads.add_attr<float>("vol",0);
        // mapping of displacement to deformation gradient
        auto& D0 = prim->quads.add_attr<zeno::vec3f>("D0");
        auto& D1 = prim->quads.add_attr<zeno::vec3f>("D1");
        auto& D2 = prim->quads.add_attr<zeno::vec3f>("D2");

        size_t nm_elms = prim->quads.size();
        for(size_t elm_id = 0;elm_id < nm_elms;++elm_id){
            auto elm = prim->quads[elm_id];
            Mat4x4d M;
            for(size_t i = 0;i < 4;++i){
                auto vert = prim->verts[elm[i]];
                M.block(0,i,3,1) << vert[0],vert[1],vert[2];
            }
            M.bottomRows(1).setConstant(1.0);
            // _elmVolume[elm_id] = fabs(M.determinant()) / 6;

            Mat3x3d Dm;
            for(size_t i = 1;i < 4;++i){
                auto vert = prim->verts[elm[i]];
                auto vert0 = prim->verts[elm[0]];
                Dm.col(i - 1) << vert[0]-vert0[0],vert[1]-vert0[1],vert[2]-vert0[2];
            }
            vol[elm_id] = Dm.determinant() / 6;

            Mat3x3d DmInv = Dm.inverse();

            D0[elm_id] = zeno::vec3f(DmInv(0,0),DmInv(0,1),DmInv(0,2));
            D1[elm_id] = zeno::vec3f(DmInv(1,0),DmInv(1,1),DmInv(1,2));
            D2[elm_id] = zeno::vec3f(DmInv(2,0),DmInv(2,1),DmInv(2,2));

            Vec3d v0;v0 << pos[elm[0]][0],pos[elm[0]][1],pos[elm[0]][2];
            Vec3d v1;v1 << pos[elm[1]][0],pos[elm[1]][1],pos[elm[1]][2];
            Vec3d v2;v2 << pos[elm[2]][0],pos[elm[2]][1],pos[elm[2]][2];
            Vec3d v3;v3 << pos[elm[3]][0],pos[elm[3]][1],pos[elm[3]][2];

            FEM_Scaler A012 = MatHelper::Area(v0,v1,v2);
            FEM_Scaler A013 = MatHelper::Area(v0,v1,v3);
            FEM_Scaler A123 = MatHelper::Area(v1,v2,v3);
            FEM_Scaler A023 = MatHelper::Area(v0,v2,v3);

            // we denote the average surface area of a tet as the characteristic norm
            cnorm[elm_id] = (A012 + A013 + A123 + A023) / 4;
        }   
        set_output("femesh",prim);
    } 
};

ZENDEFNODE(MakeFEMPrimitive, {
    {"prim",
        {"float","Stiffness","1000000"},
        {"float","VolumePreserve","0.49"},
        {"float","ExamShapeCoeff","0.0"},
        {"float","EmbedShapeCoeff","0.0"}
    },
    {"femmesh"},
    {},
    {"FEM"},
});

struct ParticlesToSegments : zeno::INode {
    virtual void apply() override {
        auto particles = get_input<zeno::PrimitiveObject>("particles");
        auto dt = get_input2<float>("dt");
        auto dir_chanel = get_param<std::string>("dir_chanel");
        const auto& ppos = particles->verts;
        if(!particles->has_attr(dir_chanel)){
            std::cout << "PARITCLES_TO_SEGMENTS NO SPECIFIED ATTRIBUTES " << dir_chanel << std::endl;
            throw std::runtime_error("PARITCLES_TO_SEGMENTS NO SPECIFIED ATTRIBUTES");
        }


        std::cout << "dt :" << dt << std::endl;
        const auto& pvel = particles->attr<zeno::vec3f>(dir_chanel);

        auto segs = std::make_shared<zeno::PrimitiveObject>();
        auto& svel = segs->add_attr<zeno::vec3f>(dir_chanel);
        segs->resize(particles->size() * 2);
        auto& segLines = segs->lines;
        segLines.resize(particles->size());
        auto& spos = segs->verts;

        for(size_t i = 0;i < particles->size();++i){
            segLines[i] = zeno::vec2i(i,i + particles->size());
            spos[i] = ppos[i];
            spos[i + particles->size()] = spos[i] + dt * pvel[i];
            svel[i] = svel[i + particles->size()] = pvel[i];
        }

        set_output("seg",segs);
    }
};

ZENDEFNODE(ParticlesToSegments, {
    {"particles",{"float","dt","1.0"}},
    {"seg"},
    {{"string","dir_chanel",""}},
    {"FEM"},
});

struct RetrieveRigidTransform : zeno::INode {
    virtual void apply() override {
        auto objRef = get_input<zeno::PrimitiveObject>("refObj");
        auto objNew = get_input<zeno::PrimitiveObject>("newObj");

        Mat4x4d refTet,newTet;
        size_t idx0 = 0;
        size_t idx1 = 1;
        size_t idx2 = 2;
        size_t idx3 = 3;

        Mat3x3d parallel_test;
        parallel_test.col(0) << objRef->verts[idx0][0],objRef->verts[idx0][1],objRef->verts[idx0][2];
        parallel_test.col(1) << objRef->verts[idx1][0],objRef->verts[idx1][1],objRef->verts[idx1][2];
        for(idx2 = idx1 + 1;idx2 < objRef->size();++idx2){
            parallel_test.col(2) << objRef->verts[idx2][0],objRef->verts[idx2][1],objRef->verts[idx2][2];
            if(fabs(parallel_test.determinant()) > 1e-6)
                break;
        }

        refTet.col(0) << parallel_test.col(0),1.0;
        refTet.col(1) << parallel_test.col(1),1.0;
        refTet.col(2) << parallel_test.col(2),1.0;
        for(idx3 = idx2 + 1;idx3 < objRef->size();++idx3){
            refTet.col(3) << objRef->verts[idx3][0],objRef->verts[idx3][1],objRef->verts[idx3][2],1.0;
            if(fabs(refTet.determinant()) > 1e-6)
                break;
        }

        newTet.col(0) << objNew->verts[idx0][0],objNew->verts[idx0][1],objNew->verts[idx0][2],1.0;
        newTet.col(1) << objNew->verts[idx1][0],objNew->verts[idx1][1],objNew->verts[idx1][2],1.0;
        newTet.col(2) << objNew->verts[idx2][0],objNew->verts[idx2][1],objNew->verts[idx2][2],1.0;
        newTet.col(3) << objNew->verts[idx3][0],objNew->verts[idx3][1],objNew->verts[idx3][2],1.0;

        // std::cout << "RETRIEVE IDX : " << idx0 << "\t" << idx1 << "\t" << idx2 << "\t" << idx3 << std::endl;

        Mat4x4d T = newTet * refTet.inverse();

        Mat3x3d R = T.block(0,0,3,3);
        Eigen::Quaternion<FEM_Scaler> quat(R);

        zeno::vec3f b(T(0,3),T(1,3),T(2,3));

        auto retb = std::make_shared<zeno::NumericObject>();
        retb->set<zeno::vec3f>(b);
        auto retq = std::make_shared<zeno::NumericObject>();
        retq->set<zeno::vec4f>(zeno::vec4f(quat.x(),quat.y(),quat.z(),quat.w()));

        auto retA = std::make_shared<MatrixObject>();
        // T = T.transpose();
        // std::cout << "T : " << std::endl << T << std::endl;
        retA->m = glm::mat4(
            T(0,0),T(0,1),T(0,2),T(3,0),
            T(1,0),T(1,1),T(1,2),T(3,1),
            T(2,0),T(2,1),T(2,2),T(3,2),
            T(0,3),T(1,3),T(2,3),T(3,3)
        );


        set_output("quat",std::move(retq));
        set_output("trans",std::move(retb));
        set_output("mat",std::move(retA));

    }
};

ZENDEFNODE(RetrieveRigidTransform,{
    {{"refObj"},{"newObj"}},
    {"quat","trans","mat"},
    {},
    {"FEM"},
});


struct AssignUniformAffineToPrim : zeno::INode {
    virtual void apply() override {
        auto rotation = get_input<zeno::NumericObject>("quat")->get<zeno::vec4f>();
        auto translate = get_input<zeno::NumericObject>("trans")->get<zeno::vec3f>();

        auto prim = get_input<zeno::PrimitiveObject>("prim");
        auto& A0s = prim->add_attr<zeno::vec3f>("A0");
        auto& A1s = prim->add_attr<zeno::vec3f>("A1");
        auto& A2s = prim->add_attr<zeno::vec3f>("A2");
        auto& A3s = prim->add_attr<zeno::vec3f>("A3");
        prim->resize(prim->size());

        glm::quat myQuat(rotation[3],rotation[0],rotation[1],rotation[2]);
        glm::mat4 R = glm::toMat4(myQuat);

        zeno::vec3f A0 = zeno::vec3f(R[0][0],R[1][0],R[2][0]);
        zeno::vec3f A1 = zeno::vec3f(R[0][1],R[1][1],R[2][1]);
        zeno::vec3f A2 = zeno::vec3f(R[0][2],R[1][2],R[2][2]);
        zeno::vec3f A3 = translate;

        std::fill(A0s.begin(),A0s.end(),A0);
        std::fill(A1s.begin(),A1s.end(),A1);
        std::fill(A2s.begin(),A2s.end(),A2);
        std::fill(A3s.begin(),A3s.end(),A3);

        set_output("prim",prim);
    }
};

ZENDEFNODE(AssignUniformAffineToPrim,{
    {{"quat"},{"trans"},{"prim"}},
    {"prim"},
    {},
    {"FEM"},
});


struct EmbedPrimitiveToVolumeMesh : zeno::INode {
    FEM_Scaler ComputeDistToTriangle(const Vec3d& vp,const Vec3d& v0,const Vec3d& v1,const Vec3d& v2){
        auto v012 = (v0 + v1 + v2) / 3;
        auto v01 = (v0 + v1) / 2;
        auto v02 = (v0 + v2) / 2;
        auto v12 = (v1 + v2) / 2;

        FEM_Scaler dist = 1e6;
        FEM_Scaler tdist = (v012 - vp).norm();
        dist = tdist < dist ? tdist : dist;
        tdist = (v01 - vp).norm();
        dist = tdist < dist ? tdist : dist;
        tdist = (v02 - vp).norm();
        dist = tdist < dist ? tdist : dist;
        tdist = (v12 - vp).norm();
        dist = tdist < dist ? tdist : dist;

        tdist = (v0 - vp).norm();
        dist = tdist < dist ? tdist : dist;
        tdist = (v1 - vp).norm();
        dist = tdist < dist ? tdist : dist;
        tdist = (v2 - vp).norm();
        dist = tdist < dist ? tdist : dist;

        return dist;
    }

    virtual void apply() override {
        auto prim = get_input<zeno::PrimitiveObject>("prim");
        auto vmesh = get_input<zeno::PrimitiveObject>("vmesh");

        auto& embed_id = prim->add_attr<float>("embed_id");
        // embed_id.resize(prim->size(),-1);
        // std::cout << "CHECK:" << embed_id[10641] << std::endl;
        auto& elm_w = prim->add_attr<zeno::vec3f>("embed_w");

        auto fitting_in = (int)get_input<zeno::NumericObject>("fitting_in")->get<float>();
        // elm_w.resize(prim->size(),zeno::vec3f(0));

        // auto& v0s = prim->add_attr<zeno::vec3f>("v0");
        // auto& v1s = prim->add_attr<zeno::vec3f>("v1");
        // auto& v2s = prim->add_attr<zeno::vec3f>("v2");
        // auto& v3s = prim->add_attr<zeno::vec3f>("v3");
        // prim->resize(prim->size());

        // std::cout << "CHECK:" << embed_id[10641] << std::endl;

        #pragma omp parallel for
        for(size_t i = 0;i < prim->size();++i){
            auto vp = Vec3d(prim->verts[i][0],prim->verts[i][1],prim->verts[i][2]);
            embed_id[i] = -1;

            FEM_Scaler closest_dist = 1e6;
            int closest_tet_id;
            Vec4d closest_tet_w;

            for(size_t j = 0;j < vmesh->quads.size();++j){
                const auto& tet = vmesh->quads[j];
                Vec3d v0 = Vec3d(vmesh->verts[tet[0]][0],vmesh->verts[tet[0]][1],vmesh->verts[tet[0]][2]);
                Vec3d v1 = Vec3d(vmesh->verts[tet[1]][0],vmesh->verts[tet[1]][1],vmesh->verts[tet[1]][2]);
                Vec3d v2 = Vec3d(vmesh->verts[tet[2]][0],vmesh->verts[tet[2]][1],vmesh->verts[tet[2]][2]);
                Vec3d v3 = Vec3d(vmesh->verts[tet[3]][0],vmesh->verts[tet[3]][1],vmesh->verts[tet[3]][2]);

                Mat4x4d M;
                M.col(0) << v0,1.0;
                M.col(1) << v1,1.0;
                M.col(2) << v2,1.0;
                M.col(3) << v3,1.0;


                auto VMT = M.determinant();

                M.col(0) << vp,1.0;
                M.col(1) << v1,1.0;
                M.col(2) << v2,1.0;
                M.col(3) << v3,1.0;

                auto VM0 = M.determinant();

                M.col(0) << v0,1.0;
                M.col(1) << vp,1.0; 
                M.col(2) << v2,1.0;
                M.col(3) << v3,1.0;

                auto VM1 = M.determinant();

                M.col(0) << v0,1.0;
                M.col(1) << v1,1.0;
                M.col(2) << vp,1.0;
                M.col(3) << v3,1.0;

                auto VM2 = M.determinant();

                M.col(0) << v0,1.0;
                M.col(1) << v1,1.0;
                M.col(2) << v2,1.0;
                M.col(3) << vp,1.0;

                auto VM3 = M.determinant();

                Vec4d w = Vec4d(VM0 / VMT,VM1 / VMT,VM2 / VMT,VM3 / VMT);

                if(w[0] > 0 && w[1] > 0 && w[2] > 0 && w[3] > 0){
                    embed_id[i] = (float)j;
                    elm_w[i][0] = w[0];
                    elm_w[i][1] = w[1];
                    elm_w[i][2] = w[2];
                    if(fabs(1 - w[0] - w[1] - w[2] - w[3]) > 1e-6){
                        std::cout << "INVALID : " << i << "\t" << j << "\t" << w.transpose() << std::endl;
                        // throw std::runtime_error("INVALID W");
                    }

                    Vec3d interpPos = w[0] * v0 + w[1] * v1 + w[2] * v2 + w[3] * v3;
                    FEM_Scaler interpError = (interpPos - vp).norm();
                    if(interpError > 1e-6){
                        std::cout << "INTERP ERROR : " << interpError << "\t" << interpPos.transpose() << "\t" << vp.transpose() << std::endl;
                    }
                    prim->verts[i] = zeno::vec3f(interpPos[0],interpPos[1],interpPos[2]);
                    // v0s[i] = zeno::vec3f(v0[0],v0[1],v0[2]);
                    // v1s[i] = zeno::vec3f(v1[0],v1[1],v1[2]);
                    // v2s[i] = zeno::vec3f(v2[0],v2[1],v2[2]);
                    // v3s[i] = zeno::vec3f(v3[0],v3[1],v3[2]);
                    
                    // if(i == 10641){
                    //     std::cout << "FIND : " << i << "\t" << j << std::endl;
                    // }
                    // std::cout << "FIND : " << i << "\t" << j << "\t" << w0 << "\t" << w1 << "\t" << w2 << "\t" << w3 << std::endl;
                    break;
                }

                if(w[0] < 0){
                    FEM_Scaler dist = ComputeDistToTriangle(vp,v1,v2,v3);
                    if(dist < closest_dist){
                        closest_dist = dist;
                        closest_tet_id = j;
                        closest_tet_w = w;
                    }
                }

                if(w[1] < 0){
                    FEM_Scaler dist = ComputeDistToTriangle(vp,v0,v2,v3);
                    if(dist < closest_dist){
                        closest_dist = dist;
                        closest_tet_id = j;
                        closest_tet_w = w;
                    }
                }

                if(w[2] < 0){
                    FEM_Scaler dist = ComputeDistToTriangle(vp,v0,v1,v3);
                    if(dist < closest_dist){
                        closest_dist = dist;
                        closest_tet_id = j;
                        closest_tet_w = w;
                    }
                }

                if(w[3] < 0){
                    FEM_Scaler dist = ComputeDistToTriangle(vp,v0,v1,v2);
                    if(dist < closest_dist){
                        closest_dist = dist;
                        closest_tet_id = j;
                        closest_tet_w = w;
                    }
                }
            }


            if(embed_id[i] < -1e-3 && fitting_in) {


                embed_id[i] = closest_tet_id;
                for(size_t i = 0;i < 4;++i)
                    closest_tet_w[i] = closest_tet_w[i] < 0 ? 0 : closest_tet_w[i];
                FEM_Scaler wsum = closest_tet_w.sum();
                closest_tet_w /= wsum;

                elm_w[i] = zeno::vec3f(closest_tet_w[0],closest_tet_w[1],closest_tet_w[2]);


                // std::cout << "CORRECT ID : " << i << "\t" << closest_tet_id << "\t" << closest_tet_w.transpose() << std::endl;
            }
        }

        for(size_t i = 0;i < embed_id.size();++i){
            if(embed_id[i] < -1e-3 && fitting_in){
                std::cerr << "COULD NOT FIND EMBED TET FOR " << i << std::endl; 
                throw std::runtime_error("COULD NOT FIND EMBED TET");
            }
        }

        // std::cout << "CHECK_ID:" << embed_id[10641] << std::endl;

        set_output("prim",prim);
    }
};

ZENDEFNODE(EmbedPrimitiveToVolumeMesh, {
    {"prim","vmesh","fitting_in"},
    {"prim"},
    {},
    {"FEM"},
});
struct InterpolateEmbedPrimitive : zeno::INode {
    virtual void apply() override {
        auto skin = get_input<zeno::PrimitiveObject>("skin");
        auto volume = get_input<zeno::PrimitiveObject>("volume");

        auto& embed_ids = skin->attr<float>("embed_id");
        auto& embed_ws = skin->attr<zeno::vec3f>("embed_w");

        auto res = std::make_shared<zeno::PrimitiveObject>(*skin);
        const auto& vposs = volume->attr<zeno::vec3f>("curPos");
        auto& eposs = res->add_attr<zeno::vec3f>("curPos");

        // const auto& v0s = res->attr<zeno::vec3f>("v0");
        // const auto& v1s = res->attr<zeno::vec3f>("v1");
        // const auto& v2s = res->attr<zeno::vec3f>("v2");
        // const auto& v3s = res->attr<zeno::vec3f>("v3");

        // #pragma omp parallel for
        for(size_t i = 0;i < skin->size();++i){
            int elm_id = (int)embed_ids[i];
            // if(fabs(elm_id - embed_ids[i]) > 0.2){
            //     std::cout << "ERROR\t" << elm_id << "\t" << embed_ids[i] << std::endl;
            // }
            auto embed_w = embed_ws[i];
            const auto& tet = volume->quads[elm_id];

            double w0 = embed_w[0];
            double w1 = embed_w[1];
            double w2 = embed_w[2];
            double w3 = 1 - w0 - w1 - w2;
            zeno::vec3f epos_copy = eposs[i];
            eposs[i] = zeno::vec3f(0);
            eposs[i] += vposs[tet[0]] * embed_w[0];
            eposs[i] += vposs[tet[1]] * embed_w[1];
            eposs[i] += vposs[tet[2]] * embed_w[2];
            eposs[i] += vposs[tet[3]] * (1 - embed_w[0] - embed_w[1] - embed_w[2]);

            // auto error_pos = zeno::length(eposs[i] - epos_copy);
            // if(error_pos > 1e-3){
            //     std::cout << "ERROR : " << i << "\t" << embed_ids[i] << "\t" \
            //         << eposs[i][0] << "\t" << eposs[i][1] << "\t" << eposs[i][2] << "\t" \
            //         <<  epos_copy[0] << "\t" << epos_copy[1] << "\t" << epos_copy[2] << std::endl;
            // }
            // eposs[i] += v0s[i] * w0;
            // eposs[i] += v1s[i] * embed_w[1];
            // eposs[i] += v2s[i] * embed_w[2];
            // eposs[i] += v3s[i] * (1 - embed_w[0] - embed_w[1] - embed_w[2]);
            // eposs[i] = w0 * v0s[i] + w1 * v1s[i] + w2 * v2s[i] + w3 * v3s[i];
        }

        set_output("res",std::move(res));
    }
};

ZENDEFNODE(InterpolateEmbedPrimitive, {
    {"skin","volume"},
    {"res"},
    {},
    {"FEM"},
});

// Get the fem elm view of mesh
struct GetTetMeshElementView : zeno::INode {
    virtual void apply() override {
        const auto& prim = get_input<zeno::PrimitiveObject>("prim");

        auto elmView = std::make_shared<zeno::PrimitiveObject>();

        auto& poss = elmView->add_attr<zeno::vec3f>("pos");
        auto& Vs = elmView->add_attr<float>("V");
        elmView->resize(prim->quads.size());
        for(size_t elm_id = 0;elm_id < elmView->size();++elm_id){
            const auto& tet = prim->quads[elm_id];
            poss[elm_id] = zeno::vec3f(0);

            Mat4x4d M;
            for(size_t i = 0;i < 4;++i){
                auto vert = prim->verts[tet[i]];
                poss[elm_id] += vert;
                M.block(0,i,3,1) << vert[0],vert[1],vert[2];
            }
            poss[elm_id] /= 4;
            M.bottomRows(1).setConstant(1.0);
            Vs[elm_id] = fabs(M.determinant()) / 6;
        }

        set_output("elmView",std::move(elmView));
    }
};

ZENDEFNODE(GetTetMeshElementView, {
    {"prim"},
    {"elmView"},
    {},
    {"FEM"},
});

struct InterpolateElmAttrib : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<zeno::PrimitiveObject>("prim");
        auto elmView = get_input<zeno::PrimitiveObject>("elmView");
        auto attr_name = get_param<std::string>("attrName");
        auto attr_type = std::get<std::string>(get_param("attrType"));

        if(!prim->has_attr(attr_name)){
            throw std::runtime_error("INPUT PRIMITIVE DOES NOT HAVE THE SPECIFIED ATTRIB");
        }
        if(elmView->size() != prim->quads.size()){
            throw std::runtime_error("THE INPUT PRIMITIVE SHOULD HAVE THE SAME NUMBER OF ELEMENTS AS THE INPUT ELMVIEW");
        }



        // using T = std::decay_t(prim->attr(attr_name)[0])

        if(prim->attr_is<float>(attr_name)){
            const auto& interp_attr = prim->attr<float>(attr_name);
            auto& elm_attr = elmView->add_attr<float>(attr_name);

            #pragma omp parallel for 
            for(size_t elm_id = 0;elm_id < elmView->size();++elm_id){
                const auto& tet = prim->quads[elm_id];
                elm_attr[elm_id] = 0;
                for(size_t i = 0;i < 4;++i){
                    elm_attr[elm_id] += interp_attr[tet[i]]/4;
                }
            }
        }else if(prim->attr_is<zeno::vec3f>(attr_name)){
            const auto& interp_attr = prim->attr<zeno::vec3f>(attr_name);
            auto& elm_attr = elmView->add_attr<zeno::vec3f>(attr_name);

            #pragma omp parallel for 
            for(size_t elm_id = 0;elm_id < elmView->size();++elm_id){
                const auto& tet = prim->quads[elm_id];
                elm_attr[elm_id] = zeno::vec3f(0);
                for(size_t i = 0;i < 4;++i){
                    elm_attr[elm_id] += interp_attr[tet[i]]/4;
                }
            }            
        }

        set_output("elmView",elmView);
    }
};

ZENDEFNODE(InterpolateElmAttrib, {
    {"prim","elmView"},
    {"elmView"},
    {{"string","attrName","RENAME ME"},{"enum float vec3f","attrType","float"}},
    {"FEM"},
});


struct EvalElmDeformationField : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<zeno::PrimitiveObject>("shape");
        auto elmView = get_input<zeno::PrimitiveObject>("shapeElmView");

        if(!prim->has_attr("curPos")){
            throw std::runtime_error("Absence of CurPos Attributes");
        }

        const auto& rvs = prim->attr<zeno::vec3f>("pos");
        const auto& dvs = prim->attr<zeno::vec3f>("curPos");

        auto& A0s = elmView->add_attr<zeno::vec3f>("A0");
        auto& A1s = elmView->add_attr<zeno::vec3f>("A1");
        auto& A2s = elmView->add_attr<zeno::vec3f>("A2");
        auto& A3s = elmView->add_attr<zeno::vec3f>("A3");

        // auto& elmPos = elmView->add_attr<zeno::vec3f>("pos");

        for(size_t elm_id = 0;elm_id < prim->quads.size();++elm_id){
            const auto& tet = prim->quads[elm_id];
            auto rv0 = rvs[tet[0]];
            auto rv1 = rvs[tet[1]];
            auto rv2 = rvs[tet[2]];
            auto rv3 = rvs[tet[3]];

            auto dv0 = dvs[tet[0]];
            auto dv1 = dvs[tet[1]];
            auto dv2 = dvs[tet[2]];
            auto dv3 = dvs[tet[3]];

            Mat4x4d rShape,dShape;
            rShape <<   rv0[0],rv1[0],rv2[0],rv3[0],
                        rv0[1],rv1[1],rv2[1],rv3[1],
                        rv0[2],rv1[2],rv2[2],rv3[2],
                             1,     1,     1,     1;

            dShape <<   dv0[0],dv1[0],dv2[0],dv3[0],
                        dv0[1],dv1[1],dv2[1],dv3[1],
                        dv0[2],dv1[2],dv2[2],dv3[2],
                             1,     1,     1,     1;
            // D = F * R
            Mat4x4d A = dShape * rShape.inverse();

            A0s[elm_id] = zeno::vec3f(A.row(0)[0],A.row(0)[1],A.row(0)[2]);
            A1s[elm_id] = zeno::vec3f(A.row(1)[0],A.row(1)[1],A.row(1)[2]);
            A2s[elm_id] = zeno::vec3f(A.row(2)[0],A.row(2)[1],A.row(2)[2]);
            A3s[elm_id] = zeno::vec3f(A.col(3)[0],A.col(3)[1],A.col(3)[2]);

            // elmPos[elm_id] = zeno::vec3f(0);
            // for(size_t i = 0;i < 4;++i)
            //     elmPos[elm_id] += dvs[tet[i]] / 4;
        }

        set_output("shapeElmView",elmView);
    }
};

ZENDEFNODE(EvalElmDeformationField, {
    {"shape","shapeElmView"},
    {"shapeElmView"},
    {},
    {"FEM"},
});

struct ExtractSurfaceMeshByTag : zeno::INode {
    virtual void apply() override {
        auto vprim = get_input<zeno::PrimitiveObject>("volume");
        auto primSurf = std::make_shared<zeno::PrimitiveObject>(*vprim);
        primSurf->tris.clear();
        primSurf->quads.clear();

        const auto& surf_tag = vprim->attr<float>("surface_tag");

        for(size_t t = 0;t < vprim->tris.size();++t){
            const auto tri = vprim->tris[t];
            if( fabs(surf_tag[tri[0]] - 1.0) < 1e-6 && 
                    fabs(surf_tag[tri[1]] - 1.0) < 1e-6 && 
                    fabs(surf_tag[tri[2]] - 1.0) < 1e-6){
                    primSurf->tris.push_back(tri);
            }
        }

        set_output("primSurf",std::move(primSurf));
    }
};

ZENDEFNODE(ExtractSurfaceMeshByTag, {
    {"volume"},
    {"primSurf"},
    {},
    {"FEM"},
});

struct AddVertID : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<zeno::PrimitiveObject>("prim");
        auto& IDs = prim->add_attr<float>("ID");
        for(size_t i = 0;i < prim->size();++i)
            IDs[i] = (float)(i);

        set_output("primOut",prim);
    }
};

ZENDEFNODE(AddVertID, {
    {"prim"},
    {"primOut"},
    {},
    {"FEM"},
});

struct ComputeNodalRotationCenter : zeno::INode {
    double ComputeSimilarity(const std::vector<double>& ws1,const std::vector<double>& ws2,double sigma){
        size_t dim = ws1.size();
        assert(ws1.size() == ws2.size());

        double w = 0;
        for(size_t i = 0;i < dim;++i)
            for(size_t j = i+1;j < dim;++j){
                auto alpha = ws1[i] + ws1[j] + ws2[i] + ws2[j];
                auto beta = ws1[i]*ws2[j] - ws1[j]*ws2[i];
                beta = beta * beta;
                w += alpha * exp(-beta/sigma/sigma);
            }
        return w;
    }

    virtual void apply() override {
        auto prim = get_input<zeno::PrimitiveObject>("prim");
        auto primNei = get_input<zeno::PrimitiveObject>("primNei");
        auto lbvh = get_input<zeno::LBvh>("lbvh");

        // auto nm_bones = get_input2<int>("nm_bones");
        auto attr_prefix = get_param<std::string>("attr_prefix");

        const auto& sigma = get_param<float>("sigma");

        const auto& pos = prim->attr<zeno::vec3f>("pos");
        const auto& npos = primNei->attr<zeno::vec3f>("pos");

        auto& rcenter = prim->add_attr<zeno::vec3f>("rCenter");


        const auto& Vs = primNei->attr<float>("V");

        // std::vec

        size_t nm_bones = 0;
        while(true){
            std::string attr_name = attr_prefix + "_" + std::to_string(nm_bones);
            if(prim->has_attr(attr_name))
                nm_bones++;
            else
                break;
        }


        #pragma omp parallel for
        for(size_t i = 0;i < prim->size();++i){
            std::vector<double> wv(nm_bones);
            for(size_t j = 0;j < nm_bones;++j){
                std::string attr_name = attr_prefix + "_" + std::to_string(j);
                wv[j] = prim->attr<float>(attr_name)[i];
            }

            rcenter[i] = zeno::vec3f(0);
            float weight_sum = 0;


            lbvh->iter_neighbors(pos[i],[&](int pid) {
                    // std::cout << "GET CALLED" << std::endl;

                    std::vector<double> wn(nm_bones);
                    for(size_t j = 0;j < nm_bones;++j){
                        std::string attr_name = attr_prefix + "_" + std::to_string(j);
                        wn[j] = primNei->attr<float>(attr_name)[pid];
                    }

                    // remove the possibly points with same location
                    float dist = zeno::length(pos[i] - npos[pid]);
                    // if(dist > 1e-6){
                        auto w = Vs[pid] * ComputeSimilarity(wv,wn,sigma);
                        weight_sum += w;
                        rcenter[i] += npos[pid] * w;

                        // if(i == 0){
                        //     std::cout << "w : " << w << "\t" << "npos:" << npos[pid][0] << "\t" << npos[pid][1] << "\t" << npos[pid][2] << "\t" \
                        //         << "wn:\t" <<  wn[0] << "\t" << wn[1] << "\t" << "wv:\t" << wv[0] << "\t" << wv[1] << std::endl;
                        // }
                    // }

                }
            );  
            if(fabs(weight_sum) == 0){
                // std::cout << "INVALID_NODE : " << i << "\t" << weight_sum << std::endl;
                rcenter[i] = pos[i];
            }
            else       
                rcenter[i] /= weight_sum;
        }

        // for(size_t i = 0;i < 100;++i)
            // std::cout << "RCENTER<" << i  << ">:\t" << rcenter[i][0] << "\t" << rcenter[i][1]  << "\t" << rcenter[i][2] << std::endl; 

        set_output("prim",prim);
    }
};


ZENDEFNODE(ComputeNodalRotationCenter, {
    {"prim","primNei","lbvh"},
    {"prim"},
    {{"string","attr_prefix","sw"},{"float","sigma","10"}},
    {"FEM"},
});


struct ComputeNodalVolume : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<zeno::PrimitiveObject>("prim");
        auto& Vs = prim->add_attr<float>("V",0);
        for(size_t elm_id = 0;elm_id < prim->quads.size();++elm_id){
            const auto& tet = prim->quads[elm_id];

            Mat4x4d M;
            for(size_t i = 0;i < 4;++i){
                auto vert = prim->verts[tet[i]];
                M.block(0,i,3,1) << vert[0],vert[1],vert[2];
            }
            M.bottomRows(1).setConstant(1.0);
            auto elmV = fabs(M.determinant()) / 6;
            for(size_t i = 0;i < 4;++i)
                Vs[tet[i]] += elmV / 4;
        }

        set_output("primOut",prim);
    }
};

ZENDEFNODE(ComputeNodalVolume, {
    {"prim"},
    {"primOut"},
    {},
    {"FEM"},
});


struct RigidTransformPrimitve : zeno::INode {
    zeno::vec4f toDualQuat(const zeno::vec4f& q,const zeno::vec3f& t){
        auto qd = zeno::vec4f(0);

        auto qx = vec(q)[0];
        auto qy = vec(q)[1];
        auto qz = vec(q)[2];
        auto qw = w(q);
        auto tx = t[0];
        auto ty = t[1];
        auto tz = t[2];

        qd[3] = -0.5*( tx*qx + ty*qy + tz*qz);          // qd.w
        qd[0] =  0.5*( tx*qw + ty*qz - tz*qy);          // qd.x
        qd[1] =  0.5*(-tx*qz + ty*qw + tz*qx);          // qd.y
        qd[2] =  0.5*( tx*qy - ty*qx + tz*qw);          // qd.z

        return qd;
    }

    zeno::vec3f vec(const zeno::vec4f& q){
        auto v = zeno::vec3f(q[0],q[1],q[2]);
        return v;
    }

    float w(const zeno::vec4f& q){
        return q[3];
    }

    zeno::vec3f transform(const zeno::vec3f& v,const zeno::vec4f& q,const zeno::vec4f& dq){
        auto d0 = vec(q);
        auto de = vec(dq);
        auto a0 = w(q);
        auto ae = w(dq);

        return v + 2*zeno::cross(d0,zeno::cross(d0,v) + a0*v) + 2*(a0*de - ae*d0 + zeno::cross(d0,de));
    }

    virtual void apply() override {
        auto prim = get_input<zeno::PrimitiveObject>("prim");
        auto q = get_input<zeno::NumericObject>("quat")->get<zeno::vec4f>();
        auto t = get_input<zeno::NumericObject>("trans")->get<zeno::vec3f>();

        // assume here the input quaternion is already normalized
        // turn the quaternion and translation into the dual parts
        auto qd = toDualQuat(q,t);


        int nv = prim->size();
        auto primOut = std::make_shared<zeno::PrimitiveObject>(*prim);
        auto& pos = primOut->attr<zeno::vec3f>("pos");

    #pragma omp parallel for
        for(int i = 0;i < nv;++i)
            pos[i] = transform(pos[i],q,qd);

        set_output("primOut",std::move(primOut));
    }

};

ZENDEFNODE(RigidTransformPrimitve, {
    {"prim","quat","trans"},
    {"primOut"},
    {},
    {"FEM"},
});


// struct ComputeExponentialWeightSimilarity : zeno::INode {
//     virtual void apply() override {
//         auto prim = get_input<zeno::PrimitiveObject>("prim");
//         auto attr_prefix = get_input2<std::string>("attrName");

//         size_t dim = 0;
//         while(true){
//             std::string attrName = attr_prefix + std::string("_") + std::to_string(dim);
//             if(has_input(attrName))
//                 dim++;
//         }

//         if(dim == 0){
//             throw std::runtime_error("NO SPECIFIED ATTRIBUTES FOUND");
//         }


//     }
// };

// ZENDEFNODE{ComputeExponentialWeightSimilarity,{
//     {"prim"},
//     {"prim"}
// }};


}