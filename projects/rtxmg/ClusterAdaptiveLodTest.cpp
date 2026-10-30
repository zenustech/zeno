#include "ClusterAdaptiveLod.h"
#include <iostream>
#include <stdexcept>

using zeno::cluster::AdaptiveLod;
static void require(bool pass, const char* message) {
    if (!pass) throw std::runtime_error(message);
}
int main() try {
    const auto hot = [](int hz) {
        AdaptiveLod p; p.reset(2.f);
        for (int i=0;i<2*hz;++i) p.advance(2,.95f,1./hz,true,true,false);
        return p.value();
    };
    require(std::abs(hot(30)/hot(120)-1.f)<.0001f,"Response depends on render FPS");
    AdaptiveLod p; p.reset(2.f);
    for(int i=0;i<120;++i)p.advance(2,.95f,1./60,true,true,false);
    const float high=p.value();
    for(int i=0;i<120;++i)p.advance(2,.4f,1./60,true,true,false);
    require(p.value()<high && p.value()>=2,"Headroom did not recover quality");
    for(int i=0;i<1800;++i)p.advance(2,.4f,1./60,true,true,false);
    require(p.value()==2,"Recovery crossed or failed to reach requested quality");
    p.reset(2);
    for(int i=0;i<100;++i)p.advance(2,.77f,1./60,true,true,false);
    require(p.value()==2,"Deadband changed quality");
    for(int i=0;i<120;++i)p.advance(2,.95f,1./60,true,true,false);
    for(int i=0;i<240;++i)p.advance(2,.77f,1./60,true,true,false);
    const float held=p.value();
    for(int i=0;i<600;++i)p.advance(2,.77f,1./60,true,true,false);
    require(p.value()==held,"Deadband never settled");
    require(p.advance(2,.99f,1,true,true,false)>held,"General pool pressure did not coarsen");
    require(p.advance(2,.99f,1,false,true,true)==2,"Fixed quality was silently relaxed");
    p.reset(2);
    for(int i=0;i<600;++i)p.advance(2,.99f,1./60,true,false,true);
    require(p.value()==2,"Pinned-only coverage drove unbounded coarsening");
    p.reset(2);
    float stable=0;
    for(int i=0;i<2400;++i) {
        const float occupancy=p.value()<4.f?.9f:.6f;
        p.advance(2,occupancy,1./60,true,true,false);
        if(i==2099)stable=p.value();
    }
    require(p.value()>=4 && p.value()<5 && p.value()==stable,"Discrete LOD cut kept oscillating");
    require(!p.pending(2,.6f,true,true,false),"Stable discrete cut keeps scheduling GPU work");
    p.viewChanged(2);
    for(int i=0;i<1800;++i)p.advance(2,.4f,1./60,true,true,false);
    require(p.value()==2,"New view inherited an obsolete recovery limit");
    require(p.advance(8,.5f,1./60,true,true,false)>=8,"Requested quality floor was violated");
    p.reset(2);
    const float pause=p.advance(2,.99f,3600,true,true,false);
    require(std::isfinite(pause)&&pause<3,"A long pause caused a quality jump");
    std::cout<<"13 adaptive LOD policy checks passed\n";
    return 0;
} catch(const std::exception& e) {
    std::cerr<<e.what()<<'\n';return 1;
}
