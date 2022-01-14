#include "declares.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/quaternion.hpp>

namespace zeno {

struct ParticlesToSegments : zeno::INode {
    virtual void apply() override {
        auto particles = get_input<zeno::PrimitiveObject>("particles");
        auto dt = get_input<zeno::NumericObject>("dt")->get<float>();
        auto attr_name = get_param<std::string>("attr_name");
        const auto& ppos = particles->verts;
        if(!particles->has_attr(attr_name)){
            std::cout << "PARITCLES_TO_SEGMENTS NO SPECIFIED ATTRIBUTES " << attr_name << std::endl;
            throw std::runtime_error("PARITCLES_TO_SEGMENTS NO SPECIFIED ATTRIBUTES");
        }

        const auto& pvel = particles->attr<zeno::vec3f>(attr_name);

        auto segs = std::make_shared<zeno::PrimitiveObject>();
        auto& svel = segs->add_attr<zeno::vec3f>(attr_name);
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
    {"particles","dt"},
    {"seg"},
    {{"string","attr_name",""}},
    {"FEM"},
});

struct RetrieveRigidTransformQuat : zeno::INode {
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
            if(fabs(parallel_test.determinant()) > 1e-8)
                break;
        }

        refTet.col(0) << parallel_test.col(0),1.0;
        refTet.col(1) << parallel_test.col(1),1.0;
        refTet.col(2) << parallel_test.col(2),1.0;
        for(idx3 = idx2 + 1;idx3 < objRef->size();++idx3){
            refTet.col(3) << objRef->verts[idx3][0],objRef->verts[idx3][1],objRef->verts[idx3][2],1.0;
            if(fabs(refTet.determinant()) > 1e-8)
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

        set_output("quat",std::move(retq));
        set_output("trans",std::move(retb));
    }
};

ZENDEFNODE(RetrieveRigidTransformQuat,{
    {{"refObj"},{"newObj"}},
    {"quat","trans"},
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
    virtual void apply() override {
        auto prim = get_input<zeno::PrimitiveObject>("prim");
        auto vmesh = get_input<zeno::PrimitiveObject>("vmesh");

        auto& embed_id = prim->add_attr<float>("embed_id");
        embed_id.resize(prim->size(),-1);

        #pragma omp parallel for
        for(size_t i = 0;i < prim->size();++i){
            auto vp = Vec3d(prim->verts[i][0],prim->verts[i][1],prim->verts[i][2]);
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

                auto w0 = VM0 / VMT;
                auto w1 = VM1 / VMT;
                auto w2 = VM2 / VMT;
                auto w3 = VM3 / VMT;

                if(w0 > 0 && w1 > 0 && w2 > 0 && w3 > 0){
                    embed_id[i] = (float)j;
                }
            }
        }

        for(size_t i = 0;i < embed_id.size();++i){
            if(embed_id[i] == -1){
                std::cerr << "COULD NOT FIND EMBED TET FOR " << i << std::endl; 
                throw std::runtime_error("COULD NOT FIND EMBED TET");
            }
        }

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
        auto prim = get_input<zeno::PrimitiveObject>("refPrim");
        auto dmesh = get_input<zeno::PrimitiveObject>("deformedMesh");
        auto rmesh = get_input<zeno::PrimitiveObject>("refMesh");

        auto& embed_id = prim->attr<float>("embed_id");

        auto res = std::make_shared<zeno::PrimitiveObject>(*prim);

        #pragma omp parallel for
        for(size_t i = 0;i < prim->size();++i){
            int elm_id = (int)embed_id[i];
            const auto& tet = dmesh->quads[elm_id];
            Mat4x4d dM,rM;
            dM.col(0) << dmesh->verts[tet[0]][0],dmesh->verts[tet[0]][1],dmesh->verts[tet[0]][2],1.0;
            dM.col(1) << dmesh->verts[tet[1]][0],dmesh->verts[tet[1]][1],dmesh->verts[tet[1]][2],1.0;
            dM.col(2) << dmesh->verts[tet[2]][0],dmesh->verts[tet[2]][1],dmesh->verts[tet[2]][2],1.0;
            dM.col(3) << dmesh->verts[tet[3]][0],dmesh->verts[tet[3]][1],dmesh->verts[tet[3]][2],1.0;

            rM.col(0) << rmesh->verts[tet[0]][0],rmesh->verts[tet[0]][1],rmesh->verts[tet[0]][2],1.0;
            rM.col(1) << rmesh->verts[tet[1]][0],rmesh->verts[tet[1]][1],rmesh->verts[tet[1]][2],1.0;
            rM.col(2) << rmesh->verts[tet[2]][0],rmesh->verts[tet[2]][1],rmesh->verts[tet[2]][2],1.0;
            rM.col(3) << rmesh->verts[tet[3]][0],rmesh->verts[tet[3]][1],rmesh->verts[tet[3]][2],1.0;

            auto A = dM * rM.inverse();

            Vec4d v;v << prim->verts[i][0],prim->verts[i][1],prim->verts[i][2],1.0;
            v = A * v;
            res->verts[i] = zeno::vec3f(v[0],v[1],v[2]);
        }

        set_output("res",std::move(res));
    }
};

ZENDEFNODE(InterpolateEmbedPrimitive, {
    {"refPrim","deformedMesh","refMesh"},
    {"res"},
    {},
    {"FEM"},
});

};