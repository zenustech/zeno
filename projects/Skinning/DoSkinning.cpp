
#include "skinning_header.h"

namespace{
using namespace zeno;

// Try to get rid of igl dependencies
struct DoSkinning : zeno::INode {
    zeno::vec4f toDualQuat(const zeno::vec4f& q,const zeno::vec3f& t) const {
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

    zeno::vec3f vec(const zeno::vec4f& q) const {
        auto v = zeno::vec3f(q[0],q[1],q[2]);
        return v;
    }

    float w(const zeno::vec4f& q) const {
        return q[3];
    }

    zeno::vec3f transform(const zeno::vec3f& v,const zeno::vec4f& q,const zeno::vec4f& dq) const {
        auto d0 = vec(q);
        auto de = vec(dq);
        auto a0 = w(q);
        auto ae = w(dq);

        return v + 2*zeno::cross(d0,zeno::cross(d0,v) + a0*v) + 2*(a0*de - ae*d0 + zeno::cross(d0,de));
    }    

    zeno::vec3f lbs_blend(const zeno::vec3f& v,
        const std::vector<zeno::vec4f>& qs,
        const std::vector<zeno::vec4f>& dqs,
        const std::vector<float>& w) const {
            auto vb = zeno::vec3f(0);
            for(size_t i = 0;i < qs.size();++i)
                vb += transform(v,qs[i],dqs[i]) * w[i];
            return vb;
    }

    zeno::vec3f dqs_blend(const zeno::vec3f& v,
        const std::vector<zeno::vec4f>& qs,
        const std::vector<zeno::vec4f>& dqs,
        const std::vector<float>& w) const {
            auto bq = zeno::vec4f(0);
            auto bdq = zeno::vec4f(0);

            for(size_t i = 0;i < qs.size();++i){
                bq += qs[i] * w[i];
                bdq += dqs[i] * w[i];
            }

            auto len = zeno::length(bq);
            bq /= len;
            bdq /= len;

            return transform(v,bq,bdq);
    }

    zeno::vec3f cors_blend(const zeno::vec3f& v,
        const std::vector<zeno::vec4f>& qs,
        const std::vector<zeno::vec4f>& dqs,
        const zeno::vec3f& rc,
        const std::vector<float>& w) const {
            auto brc = zeno::vec3f(0);
            auto bq = zeno::vec4f(0);
            for(size_t i = 0;i < qs.size();++i){
                bq += qs[i] * w[i];
                brc += transform(rc,qs[i],dqs[i]) * w[i];
            }
            auto len = zeno::length(bq);
            bq /= len;
             
            auto bt = transform(rc,bq,zeno::vec4f(0));
            bt = brc - bt;

            auto bdq = toDualQuat(bq,bt);

            return transform(v,bq,bdq);
    }

    virtual void apply() override {
        auto shape = get_input<PrimitiveObject>("shape");
        auto algorithm = std::get<std::string>(get_param("algorithm"));
        auto attr_prefix = get_param<std::string>("attr_prefix");
        auto outputChannel = get_param<std::string>("out_channel");  

        auto qs_ = get_input<zeno::ListObject>("Qs")->get<std::shared_ptr<NumericObject>>();
        auto ts_ = get_input<zeno::ListObject>("Ts")->get<std::shared_ptr<NumericObject>>(); 

        size_t nm_handles = 0;
        while(true){
            std::string attr_name = attr_prefix + "_" + std::to_string(nm_handles);
            if(shape->has_attr(attr_name)){
                nm_handles++;
                continue;
            }
            break;
        }

        std::vector<zeno::vec4f> qs(nm_handles),dqs(nm_handles);  
        std::vector<std::string> attr_names(nm_handles);
    #pragma omp parallel for if(nm_handles > 1000)
        for(int i = 0;i < nm_handles;++i){
            qs[i] = qs_[i]->get<zeno::vec4f>();
            dqs[i] = toDualQuat(qs[i],ts_[i]->get<zeno::vec3f>());
            attr_names[i] = attr_prefix +  "_" + std::to_string(i);
        }

        auto& out = shape->add_attr<zeno::vec3f>(outputChannel);
        const auto& pos = shape->attr<zeno::vec3f>("pos");
        size_t nv = shape->size();
    #pragma omp parallel for if (nv > 10000)
        for(size_t i = 0;i < nv;++i){
            std::vector<float> w(nm_handles);
            for(size_t j = 0;j < nm_handles;++j)
                w[j] = shape->attr<float>(attr_names[j])[i];
            if(algorithm == "DQS")
                out[i] = dqs_blend(pos[i],qs,dqs,w);
            else if(algorithm == "LBS")
                out[i] = lbs_blend(pos[i],qs,dqs,w);
            else if(algorithm == "CoRs")
                out[i] = cors_blend(pos[i],qs,dqs,shape->attr<zeno::vec3f>("rCenter")[i],w);
        }

        set_output("shape",shape);
    }
};


ZENDEFNODE(DoSkinning, {
    {"shape","Qs","Ts"},
    {"shape"},
    {{"enum LBS DQS CoRs","algorithm","DQS"},{"string","attr_prefix","sw"},{"string","out_channel","curPos"}},
    {"Skinning"},
});

};