#include "../../Structures.hpp"
#include "zensim/Logger.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/geometry/PoissonDisk.hpp"
#include "zensim/geometry/VdbLevelSet.h"
#include "zensim/geometry/VdbSampler.h"
#include "zensim/io/MeshIO.hpp"
#include "zensim/math/bit/Bits.h"
#include "zensim/math/Vec.h"
#include "zensim/types/Property.h"
#include <atomic>
#include <zeno/VDBGrid.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>


namespace zeno {

struct ZSDoSkinning : INode {
    using T = float;
    using vec3 = zs::vec<T,3>;
    using vec4 = zs::vec<T,4>;

    constexpr vec3 vec(const vec4& q) {
        return vec3{q[0],q[1],q[2]};
    }

    constexpr T w(const vec4& q) {
        return q[3];
    }

    constexpr vec4 dual_quat(const vec4& q,const vec3& t) {
        auto qd = vec4::zeros();

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

    constexpr vec3 transform(const vec3& v,const vec4& q,const vec4& dq){
        auto d0 = vec(q);
        auto de = vec(dq);
        auto a0 = w(q);
        auto ae = w(dq);

        return v + 2*cross(d0,cross(d0,v) + a0*v) + 2*(a0*de - ae*d0 + cross(d0,de));
    }

    // template<typename PoseBuffer,typename WeightBuffer>
    // constexpr vec3 lbs_blend(const vec3& v,
    //     const PoseBuffer& pose_buffer,
    //     const WeightBuffer& weight_buffer,
    //     const zs::SmallString& prefix) {
    //         auto bv = vec3::zeros();
    //         // int nm_bp = weight_buffer.size();
    //         for(int i = 0;i != weight_buffer.size();++i){
    //             auto idx = reinterpret_bits<int>(weight_buffer("inds",i));
    //             auto w = weight_buffer(prefix,i);
    //             vb += transform(v,pose_buffer.pack<4>("q",idx),pose_buffer.pack<4>("dq",idx)) * w;
    //         }
    //         return bv;
    // }

    // template<typename PoseBuffer, typename WeightBuffer>
    // constexpr vec3 dqs_blend(const vec3& v,
    //     const PoseBuffer& pose_buffer,
    //     const WeightBuffer& weight_buffer,
    //     const zs::SmallString& prefix) {
    //         auto bq = vec4::zeros();
    //         auto dbq = vec4::zeros();

    //         for(int i = 0;i != weight_buffer.size();++i){
    //             auto idx = reinterpret_bits<int>(weight_buffer("inds",i));
    //             auto w = weight_buffer(prefix,i);
    //             bq  += pose_buffer.pack<4>("q",idx)  * w[i];
    //             bdq += pose_buffer.pack<4>("dq",idx) * w[i];
    //         }

    //         auto len = bq.length();
    //         bq /= len;
    //         bdq /= len;

    //         return transform(v,bq,bdq);
    // }

    // template<typename PoseBuffer, typename WeightBuffer>
    // constexpr vec3 cors_blend(const vec3& v,
    //     const vec3& rc,
    //     const PoseBuffer& pose_buffer,
    //     const WeightBuffer& weight_buffer,
    //     const zs::SmallString& prefix) {
    //         auto brc = vec3::zeros();
    //         auto bq = vec4::zeros();
    //         for(int i = 0;i < weight_buffer.size();++i){
    //             auto idx = reinterpret_bits<int>(weight_buffer("inds",i));
    //             auto w = weight_buffer(prefix,i);

    //             auto q = pose_buffer.pack<4>("q",idx);
    //             auto dq = pose_buffer.pack<4>("dq",idx);
    //             bq += q * w[i];
    //             brc += transform(rc,q,dq) * w;
    //         }

    //         auto len = bq.length();
    //         bq /= len;

    //         auto bt = transform(rc,bq,vec4::zeros());
    //         bt = brc - bt;

    //         auto bdq = dual_quat(bq,bt);

    //         return transform(v,bq,dbq);
    // }

    virtual void apply() override {
        using namespace zs;
        auto zspars = get_input<ZenoParticles>("ZSParticles");
        auto algorithm = std::get<std::string>(get_param("algorithm"));
        auto prefix = get_param<std::string>("weight_channel");
        auto inAttr = get_param<std::string>("inAttr");
        auto outAttr = get_param<std::string>("outAttr");

        auto qs_ = get_input<zeno::ListObject>("Qs")->get<std::shared_ptr<NumericObject>>();
        auto ts_ = get_input<zeno::ListObject>("Ts")->get<std::shared_ptr<NumericObject>>();

        if(qs_.size() != ts_.size())
            throw std::runtime_error("the size of qs and ts do not match");

        int nm_handles = qs_.size();

        auto& pars = zspars->getParticles();
        if(!pars.hasProperty(prefix))
            throw std::runtime_error("NO SPECIFIED WEIGHT CHANNEL FOUND");
        if(!pars.hasProperty("inds"))
            throw std::runtime_error("NO WEIGHT INDICES FOUND");


        std::vector<zs::PropertyTag> tags{{"q",4},{"dq",4}};
        auto pose_buffer = typename ZenoParticles::particles_t(tags,nm_handles,zs::memsrc_e::host);

        // std::vector<zeno::vec4f> qs(nm_handles),dqs(nm_handles);  
        // std::vector<std::string> attr_names(nm_handles);

        

        constexpr auto space = execspace_e::openmp;
        auto ompExec = omp_exec();
        
        auto weight_dim = pars.getChannelSize(prefix);
        fmt::print("weight_dim : {}\n",weight_dim);
        fmt::print("nm_handles : {}\n",nm_handles);
        
        // transform the quaternion + trans into quaternion + dual quaternion pairs
        ompExec(zs::Collapse(nm_handles),
            [this,qs_,ts_,pose_buffer = proxy<space>({},pose_buffer)] (int hi) mutable{
                auto zq = qs_[hi]->get<zeno::vec4f>();
                auto zt = ts_[hi]->get<zeno::vec3f>();
                pose_buffer.tuple<4>("q",hi) = vec4{zq[0],zq[1],zq[2],zq[3]};
                pose_buffer.tuple<4>("dq",hi) = dual_quat(vec4{zq[0],zq[1],zq[2],zq[3]},vec3{zt[0],zt[1],zt[2]});
        });

        constexpr auto cspace = execspace_e::cuda;
        auto cudaPol = cuda_exec();

        pose_buffer = pose_buffer.clone({zs::memsrc_e::device, 0});

        cudaPol(zs::range(pars.size()),
            [this,pars = proxy<cspace>({},pars),pose_buffer = proxy<cspace>({},pose_buffer),prefix = zs::SmallString(prefix),
                    alg = zs::SmallString(algorithm),inAttr = zs::SmallString(inAttr),outAttr = zs::SmallString(outAttr)]
                ZS_LAMBDA(int vi) mutable {
                    auto v = pars.pack<3>(inAttr,vi);
                    if(alg == "LBS"){
                        pars.tuple<3>(outAttr,vi) = vec3::zeros();
                        // int nm_bp = weight_buffer.size();
                        for(int i = 0;i != pars.propertySize(prefix);++i){
                            auto idx = reinterpret_bits<int>(pars("inds",i,vi));
                            auto w = pars(prefix,i,vi);
                            pars.tuple<3>(outAttr,vi) = pars.pack<3>(outAttr,vi) +  transform(v,pose_buffer.pack<4>("q",idx),pose_buffer.pack<4>("dq",idx)) * w;
                        }
                    }
                    else if(alg == "DQS"){
                        auto bq = vec4::zeros();
                        auto bdq = vec4::zeros();

                        for(int i = 0;i != pars.propertySize(prefix);++i){
                            auto idx = reinterpret_bits<int>(pars("inds",i,vi));
                            auto w = pars(prefix,i,vi);
                            bq  += pose_buffer.template pack<4>("q",idx)  * w;
                            bdq += pose_buffer.template pack<4>("dq",idx) * w;
                        }

                        auto len = bq.length();
                        bq /= len;
                        bdq /= len;

                        pars.tuple<3>(outAttr,vi) = transform(v,bq,bdq);
                    }
                    else if(alg == "CoRs"){
                        auto rc = pars.pack<3>("rc",vi);
                        auto brc = vec3::zeros();
                        auto bq = vec4::zeros();
                        for(int i = 0;i < pars.propertySize(prefix);++i){
                            auto idx = reinterpret_bits<int>(pars("inds",i,vi));
                            auto w = pars(prefix,i,vi);

                            auto q = pose_buffer.pack<4>("q",idx);
                            auto dq = pose_buffer.pack<4>("dq",idx);
                            bq += q * w;
                            brc += transform(rc,q,dq) * w;
                        }

                        auto len = bq.length();
                        bq /= len;

                        auto bt = transform(rc,bq,vec4::zeros());
                        bt = brc - bt;

                        auto bdq = dual_quat(bq,bt);

                        pars.tuple<3>(outAttr,vi) = transform(v,bq,bdq);
                    }
        });

        set_output("ZSParticles",get_input<ZenoParticles>("ZSParticles"));
    }
};

ZENDEFNODE(ZSDoSkinning, {
    {"ZSParticles","Qs","Ts"},
    {"ZSParticles"},
    {{"enum LBS DQS CoRs","algorithm","DQS"},{"string","weight_channel","sw"},{"string","inAttr","x"},{"string","outAttr","x"}},
    {"ZSGeometry"},
});


};