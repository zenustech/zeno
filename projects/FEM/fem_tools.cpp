#include "declares.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/quaternion.hpp>

namespace zeno {

struct MatrixObject : zeno::IObject{
    std::variant<glm::mat3, glm::mat4> m;
};

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
            if(fabs(parallel_test.determinant()) > 1e-3)
                break;
        }

        refTet.col(0) << parallel_test.col(0),1.0;
        refTet.col(1) << parallel_test.col(1),1.0;
        refTet.col(2) << parallel_test.col(2),1.0;
        for(idx3 = idx2 + 1;idx3 < objRef->size();++idx3){
            refTet.col(3) << objRef->verts[idx3][0],objRef->verts[idx3][1],objRef->verts[idx3][2],1.0;
            if(fabs(refTet.determinant()) > 1e-3)
                break;
        }

        newTet.col(0) << objNew->verts[idx0][0],objNew->verts[idx0][1],objNew->verts[idx0][2],1.0;
        newTet.col(1) << objNew->verts[idx1][0],objNew->verts[idx1][1],objNew->verts[idx1][2],1.0;
        newTet.col(2) << objNew->verts[idx2][0],objNew->verts[idx2][1],objNew->verts[idx2][2],1.0;
        newTet.col(3) << objNew->verts[idx3][0],objNew->verts[idx3][1],objNew->verts[idx3][2],1.0;

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
        std::cout << "T : " << std::endl << T << std::endl;
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
        // elm_w.resize(prim->size(),zeno::vec3f(0));

        // auto& v0s = prim->add_attr<zeno::vec3f>("v0");
        // auto& v1s = prim->add_attr<zeno::vec3f>("v1");
        // auto& v2s = prim->add_attr<zeno::vec3f>("v2");
        // auto& v3s = prim->add_attr<zeno::vec3f>("v3");
        // prim->resize(prim->size());

        // std::cout << "CHECK:" << embed_id[10641] << std::endl;

        #pragma omp parallel for
        for(intptr_t i = 0;i < prim->size();++i){
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


            if(embed_id[i] < -1e-3) {


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
            if(embed_id[i] < -1e-3){
                std::cerr << "COULD NOT FIND EMBED TET FOR " << i << std::endl; 
                throw std::runtime_error("COULD NOT FIND EMBED TET");
            }
        }

        // std::cout << "CHECK_ID:" << embed_id[10641] << std::endl;

        set_output("prim",prim);
    }
};

ZENDEFNODE(EmbedPrimitiveToVolumeMesh, {
    {"prim","vmesh"},
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
        for(intptr_t i = 0;i < skin->size();++i){
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
            for(intptr_t elm_id = 0;elm_id < elmView->size();++elm_id){
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
            for(intptr_t elm_id = 0;elm_id < elmView->size();++elm_id){
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

        // for(size_t elm_id = 0;elm_id < vprim->quads.size();++elm_id){
        //     const auto& tet = vprim->quads[elm_id];
        //     //0,1,2;1,2,3;2,3,0;3,0,1
        //     for(size_t i = 0;i < 4;++i){
        //         auto tris = zeno::vec3i(tet[i],tet[(i+1)%4],tet[(i+2)%4]);
        //         if( fabs(surf_tag[tris[0]] - 1.0) < 1e-6 && 
        //             fabs(surf_tag[tris[1]] - 1.0) < 1e-6 && 
        //             fabs(surf_tag[tris[2]] - 1.0) < 1e-6){
        //             primSurf->tris.push_back(tris);
        //         }
        //     }
        // }

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



}
