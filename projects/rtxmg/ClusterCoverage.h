#pragma once
#include "ClusterCache.h"
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace zeno::cluster::coverage {
#ifdef __CUDACC__
#define ZENO_COVERAGE_HD __host__ __device__
#else
#define ZENO_COVERAGE_HD
#endif
struct Point {
    double x[3]{};
    ZENO_COVERAGE_HD Point operator+(const Point& p) const {return {{x[0]+p.x[0],x[1]+p.x[1],x[2]+p.x[2]}};}
    ZENO_COVERAGE_HD Point operator-(const Point& p) const {return {{x[0]-p.x[0],x[1]-p.x[1],x[2]-p.x[2]}};}
    ZENO_COVERAGE_HD Point operator*(double s) const {return {{x[0]*s,x[1]*s,x[2]*s}};}
};
ZENO_COVERAGE_HD inline double dot(Point a,Point b){return a.x[0]*b.x[0]+a.x[1]*b.x[1]+a.x[2]*b.x[2];}
struct Triangle {Point p[3];};
inline Triangle triangle(const Vertex* const v[3]) {
    Triangle t;for(int i=0;i<3;++i)for(int j=0;j<3;++j)t.p[i].x[j]=v[i]->position[j];return t;
}
ZENO_COVERAGE_HD inline double segmentDistance(Point p,Point a,Point b) {
    const auto ab=b-a;const double d=dot(ab,ab);
    const auto q=a+ab*(d>0?fmin(fmax(dot(p-a,ab)/d,0.),1.):0.);
    return dot(p-q,p-q);
}
ZENO_COVERAGE_HD inline double triangleDistance(Point p,const Triangle& t) {
    const auto a=t.p[0],b=t.p[1],c=t.p[2],ab=b-a,ac=c-a,ap=p-a;
    const double aa=dot(ab,ab),bb=dot(ac,ac),cross=dot(ab,ac),det=aa*bb-cross*cross;
    if(det>DBL_EPSILON*aa*bb) {
        const double u=(dot(ap,ab)*bb-dot(ap,ac)*cross)/det,
                     v=(dot(ap,ac)*aa-dot(ap,ab)*cross)/det;
        if(u>=0&&v>=0&&u+v<=1){const auto d=p-(a+ab*u+ac*v);return dot(d,d);}
    }
    return fmin(segmentDistance(p,a,b),fmin(segmentDistance(p,b,c),segmentDistance(p,c,a)));
}
#undef ZENO_COVERAGE_HD
// Small offline BVH. Geometry and original read-only payloads are never changed.
class Surface {
public:
    struct Node {Point lo,hi;uint32_t begin{},end{},left{},right{};};
private:
    std::vector<Triangle> triangles;
    std::vector<Node> nodes;
    uint32_t build(uint32_t begin,uint32_t end) {
        Node n;n.begin=begin;n.end=end;
        for(int j=0;j<3;++j){n.lo.x[j]=std::numeric_limits<double>::infinity();n.hi.x[j]=-n.lo.x[j];}
        for(uint32_t i=begin;i<end;++i)for(const auto& p:triangles[i].p)for(int j=0;j<3;++j){n.lo.x[j]=std::min(n.lo.x[j],p.x[j]);n.hi.x[j]=std::max(n.hi.x[j],p.x[j]);}
        const uint32_t id=uint32_t(nodes.size());nodes.push_back(n);
        if(end-begin>8) {
            int axis=0;for(int j=1;j<3;++j)if(n.hi.x[j]-n.lo.x[j]>n.hi.x[axis]-n.lo.x[axis])axis=j;
            const uint32_t mid=begin+(end-begin)/2;
            std::nth_element(triangles.begin()+begin,triangles.begin()+mid,triangles.begin()+end,[axis](const Triangle& a,const Triangle& b){
                return a.p[0].x[axis]+a.p[1].x[axis]+a.p[2].x[axis]<b.p[0].x[axis]+b.p[1].x[axis]+b.p[2].x[axis];});
            const auto left=build(begin,mid),right=build(mid,end);nodes[id].left=left;nodes[id].right=right;
        }
        return id;
    }
    double box(Point p,uint32_t id) const {
        double d=0;for(int j=0;j<3;++j){const auto& n=nodes[id];const double v=std::max({n.lo.x[j]-p.x[j],0.,p.x[j]-n.hi.x[j]});d+=v*v;}return d;
    }
    void nearest(Point p,uint32_t id,double& best,uint32_t& closest)const {
        const auto& n=nodes[id];if(box(p,id)>=best)return;
        if(n.left) {
            const double a=box(p,n.left),b=box(p,n.right);
            if(a<b){nearest(p,n.left,best,closest);nearest(p,n.right,best,closest);}else{nearest(p,n.right,best,closest);nearest(p,n.left,best,closest);}
        } else for(uint32_t i=n.begin;i<n.end;++i){const double d=triangleDistance(p,triangles[i]);if(d<best){best=d;closest=i;}}
    }
public:
    const std::vector<Triangle>& data()const {return triangles;}
    const std::vector<Node>& hierarchy()const {return nodes;}
    explicit Surface(std::vector<Triangle> t):triangles(std::move(t)) {
        if(triangles.empty())throw std::runtime_error("Empty coarse coverage surface");
        nodes.reserve(triangles.size()/2+1);build(0,uint32_t(triangles.size()));
    }
    double closest(Point p,uint32_t& triangle)const {double d=std::numeric_limits<double>::infinity();triangle=0;nearest(p,0,d,triangle);return std::sqrt(d);}
    double distance(Point p)const {uint32_t id;return closest(p,id);}
    // One convex coarse triangle also bounds the entire fine triangle by its
    // three vertex distances. This cheaper, more conservative mode needs only
    // one BVH query and is practical for whole-asset certification.
    double convexBound(const Triangle& t)const {
        uint32_t id;closest((t.p[0]+t.p[1]+t.p[2])*(1./3),id);
        double b=0;for(const auto& p:t.p)b=std::max(b,triangleDistance(p,triangles[id]));
        return std::sqrt(b);
    }
    // Seven samples include vertices, edge midpoints and centroid. Subdivision
    // at the midpoints gives subtriangles of maximum edge L/2. Every point of
    // such a triangle is within (2/3)*(L/2) of a sampled vertex. Distance to a
    // surface is 1-Lipschitz, so max(sample distances)+L/3 bounds the entire
    // directed fine-to-coarse distance, including disappearing thin components.
    double bound(const Triangle& t)const {
        double d=0,edge=0;uint32_t candidates[7],count=0;
        auto sample=[&](Point p){uint32_t id;d=std::max(d,closest(p,id));candidates[count++]=id;};
        for(int i=0;i<3;++i){sample(t.p[i]);sample((t.p[i]+t.p[(i+1)%3])*.5);
            const auto e=t.p[i]-t.p[(i+1)%3];edge=std::max(edge,dot(e,e));}
        sample((t.p[0]+t.p[1]+t.p[2])*(1./3));
        double result=d+std::sqrt(edge)/3;
        // Distance to one convex triangle is a convex function. Its maximum
        // over the fine triangle is bounded by the maximum at the three fine
        // vertices. Every candidate gives a second valid upper bound, often
        // zero when the simplifier only regrouped unchanged surface triangles.
        for(uint32_t i=0;i<count;++i) {
            bool repeated=false;for(uint32_t j=0;j<i;++j)repeated|=candidates[i]==candidates[j];if(repeated)continue;
            double bound=0;for(const auto& p:t.p)bound=std::max(bound,triangleDistance(p,triangles[candidates[i]]));
            result=std::min(result,std::sqrt(bound));
        }
        return result;
    }
};
}
