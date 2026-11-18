#include "ClusterCoverage.h"
#include <iostream>
using namespace zeno::cluster::coverage;
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
int main()try {
    Triangle plane;plane.p[0]={{0,0,0}};plane.p[1]={{2,0,0}};plane.p[2]={{0,2,0}};
    Surface flat({plane});
    require(flat.bound(plane)<1e-12,"Unchanged triangles must not force refinement");
    require(flat.convexBound(plane)<1e-12,"Unchanged convex triangles must not force refinement");
    require(std::abs(flat.distance({{.2,.2,3}})-3)<1e-12,"Triangle plane projection");
    require(std::abs(flat.distance({{3,0,0}})-1)<1e-12,"Triangle edge distance");
    const Triangle degenerate{};
    Surface point({degenerate});require(std::abs(point.distance({{3,4,0}})-5)<1e-12,"Degenerate point surface");
    Triangle missingBlade;missingBlade.p[0]={{10,0,0}};missingBlade.p[1]={{10,.02,0}};missingBlade.p[2]={{10,0,.3}};
    require(flat.bound(missingBlade)>=8,"Disconnected blade must contribute coverage error");
    require(flat.convexBound(missingBlade)>=8,"Convex bound lost a disconnected blade");
    // Parallel fine/coarse triangles have exactly one unit surface distance.
    auto raised=plane;for(auto& p:raised.p)p.x[2]=1;
    require(std::abs(flat.bound(raised)-1)<1e-12,"Parallel displacement bound should be tight");
    require(std::abs(flat.convexBound(raised)-1)<1e-12,"Parallel convex bound should be tight");
    // Check the continuous-surface bound against densely sampled barycentric
    // points, including a triangle whose vertices alone miss an interior gap.
    const double bound=point.bound(plane);
    for(int i=0;i<=64;++i)for(int j=0;j<=64-i;++j) {
        auto p=plane.p[0]*(1.-double(i+j)/64)+plane.p[1]*(double(i)/64)+plane.p[2]*(double(j)/64);
        require(point.distance(p)<=bound+1e-12,"Coverage bound underestimated triangle interior");
        require(point.distance(p)<=point.convexBound(plane)+1e-12,"Convex bound underestimated triangle interior");
    }
    std::vector<Triangle> tiles;
    for(int i=0;i<64;++i){auto t=plane;for(auto& p:t.p)p.x[0]+=i*3;tiles.push_back(t);}
    Surface bvh(tiles);std::reverse(tiles.begin(),tiles.end());Surface reversed(tiles);
    for(int i=0;i<191;++i){Point p{{i*.99,.2,.37}};double brute=std::numeric_limits<double>::infinity();
        for(const auto& t:tiles)brute=std::min(brute,triangleDistance(p,t));
        require(std::abs(bvh.distance(p)-std::sqrt(brute))<1e-10,"BVH pruning changed nearest surface");
        require(std::abs(bvh.distance(p)-reversed.distance(p))<1e-10,"BVH input-order dependence");}
    std::cout<<"Surface coverage tests passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
