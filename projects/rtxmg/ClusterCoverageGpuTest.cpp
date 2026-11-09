#include "ClusterCoverageGpu.h"
#include <iostream>
#include <random>
using namespace zeno::cluster::coverage;
int main()try {
    std::mt19937 rng(1234);std::uniform_real_distribution<double> random(-5,5);
    GpuBatch gpu,convexGpu(true);std::vector<double> expected(128),actual(128),convexExpected(128),convexActual(128);
    for(uint32_t group=0;group<128;++group) {
        std::vector<Triangle> coarse,fine;
        for(int i=0;i<32;++i){Triangle t;for(auto& p:t.p)for(auto& x:p.x)x=random(rng);
            if(group%8==0)for(auto& p:t.p)p.x[2]=0;
            if(group%16==0)t.p[1]=t.p[2]=t.p[0];
            coarse.push_back(t);}
        Surface s(coarse);
        for(int i=0;i<96;++i){auto t=coarse[i%coarse.size()];
            if(i>=32)for(auto& p:t.p)for(auto& x:p.x)x+=random(rng)*.1;
            fine.push_back(t);expected[group]=std::max(expected[group],s.bound(t));
            convexExpected[group]=std::max(convexExpected[group],s.convexBound(t));}
        gpu.add(group,s,fine);
        convexGpu.add(group,s,fine);
    }
    gpu.flush(actual);
    convexGpu.flush(convexActual);
    for(size_t i=0;i<actual.size();++i)if(std::abs(actual[i]-expected[i])>1e-10*std::max(1.,expected[i]))
        throw std::runtime_error("CPU/GPU coverage bounds disagree");
    for(size_t i=0;i<actual.size();++i) {
        if(std::abs(convexActual[i]-convexExpected[i])>1e-10*std::max(1.,convexExpected[i]))
            throw std::runtime_error("CPU/GPU convex coverage bounds disagree");
        if(convexActual[i]+1e-10<actual[i])throw std::runtime_error("Convex bound must enclose tight bound");
    }
    // A second flush exercises buffer reuse and independent group ownership.
    Triangle t;t.p[1]={{1,0,0}};t.p[2]={{0,1,0}};Surface flat({t});
    gpu.add(127,flat,{t});gpu.flush(actual);
    if(actual[127]!=0)throw std::runtime_error("Unchanged GPU surface forced refinement");
    std::cout<<"CPU/GPU coverage equivalence passed for random, coplanar and degenerate surfaces\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
