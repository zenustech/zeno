#include <cctype>
#include <filesystem>
#include <sstream>
#include <fstream>
#include <zeno/core/Graph.h>
#include <zeno/extra/assetDir.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/utils/log.h>
#include <zeno/utils/vec.h>
#include <zeno/zeno.h>
#include <string>

namespace zeno {

struct SelectSelfIntersectingFaces : INode {
    class BBox {
        public:
        vec3f min, max;
        BBox() : min(std::numeric_limits<float>::max()), max(-std::numeric_limits<float>::max()) {}
        BBox(const std::vector<vec3f>& points) {
            min = vec3f(std::numeric_limits<float>::max());
            max = vec3f(-std::numeric_limits<float>::max());
            for (auto &it: points) {
                for (int i = 0; i < 3; ++i) {
                    if (it[i] < min[i])
                        min[i] = it[i];
                    if (it[i] > max[i])
                        max[i] = it[i];
                }
            }
        }

        void add(const BBox& bbox) {
            for (int i = 0; i < 3; ++i) {
                if (bbox.min[i] < min[i])
                    min[i] = bbox.min[i];
                if (bbox.max[i] > max[i])
                    max[i] = bbox.max[i];
            }
        }

        float diag() {
            if (max[0] < min[0] || max[1] < min[1] || max[2] < min[2])
                return 0.f;
            return distance(max, min); 
        }

        bool collide(const BBox& b) const {
            return b.min[0]<max[0] && b.max[0]>min[0] &&
                b.min[1]<max[1] && b.max[1]>min[1] &&
                b.min[2]<max[2] && b.max[2]>min[2];
        }

        void intersect(const BBox & b) {
            if(min[0] < b.min[0]) min[0] = b.min[0];
            if(min[1] < b.min[1]) min[1] = b.min[1];
            if(min[2] < b.min[2]) min[2] = b.min[2];

            if(max[0] > b.max[0]) max[0] = b.max[0];
            if(max[1] > b.max[1]) max[1] = b.max[1];
            if(max[2] > b.max[2]) max[2] = b.max[2];

            if(min[0]>max[0] || min[1]>max[1] || min[2]>max[2]) {
                min = vec3f(std::numeric_limits<float>::max());
                max = vec3f(-std::numeric_limits<float>::max());
            }
        }
    };

    class MeshGrid {
        #define EPSIL 0.000001
        public:
        BBox bbox{};
        vec3i siz{};
        vec3f voxel{}, dim{};
		std::vector<Link> links;
		std::vector<Link*> grid;

        MeshGrid(const AttrVector<vec3i>& faces, const AttrVector<vec3f>& verts) {
			for(auto &fi : faces) {
                std::vector<vec3f> points{};
                for (int i = 0; i < 3; ++i)
                    points.push_back(verts[fi[i]]);
				bbox.add(BBox(points));
			}
			int size = faces->size();

			///inflate the bb calculated
			float infl = bbox.diag() / size;
            vec3f dim;
            bbox.min = bbox.min - infl;
            bbox.max = bbox.max + infl;
            dim = bbox.max - bbox.min;
            siz = vec3f(1);

            double eps  = normalize(dim) * 1e-4;
            int ncell = std::max(1, size);
            if(dim[0] > eps) {
                if(dim[1] > eps) {
                    if(dim[2] > eps) {
                        double k = std::pow((double)(ncell/(dim[0]*dim[1]*dim[2])), double(1.0/3.f));
                        siz[0] = int(dim[0] * k);
                        siz[1] = int(dim[1] * k);
                        siz[2] = int(dim[2] * k);
                    } else {
                        siz[0] = int(std::sqrt(ncell*dim[0]/dim[1]));
                        siz[1] = int(std::sqrt(ncell*dim[1]/dim[0]));
                    }
                } else {
                    if(dim[2] > eps) {
                        siz[0] = int(std::sqrt(ncell*dim[0]/dim[2]));
                        siz[2] = int(std::sqrt(ncell*dim[2]/dim[0]));
                    } else
                        siz[0] = int(ncell);
                }
            } else {
                if(dim[1] > eps) {
                    if(dim[2] > eps) {
                        siz[1] = int(std::sqrt(ncell*dim[1]/dim[2]));
                        siz[2] = int(std::sqrt(ncell*dim[2]/dim[1]));
                    } else
                        siz[1] = int(ncell);
                } else if(dim[2] > eps)
                    siz[2] = int(ncell);
            }
            for (int i = 0; i < 3; ++i)
                siz[i] = std::max(siz[i],1);
			voxel = dim / siz;
			
			grid.resize(siz[0]*siz[1]*siz[2]+1);
            links.clear();
            for (int f = 0; f < faces->size(); ++f) {
                auto fi = faces[f];
                std::vector<vec3f> points{};
                for (int j = 0; j < 3; ++j)
                    points.push_back(verts[fi[j]]);
                BBox bb{points};
                bb.intersect(bbox);
                if(bb.diag() > 0) {
                    BBox ib;
                    this->BoxToIBox( bb,ib );
                    for(int z=ib.min[2];z<=ib.max[2];++z) {
                        int bz = z*siz[1];
                        for(int y=ib.min[1];y<=ib.max[1];++y) {
                            int by = (y+bz)*siz[0];
                            for(int x=ib.min[0];x<=ib.max[0];++x)
                                // Inserire calcolo cella corrente
                                // if( pt->Intersect( ... )
                                links.push_back( Link(&(*i),by+x) );
                        }
                    }
                }
            }

            links.push_back(Link(NULL, int(grid.size())-1));

            sort( links.begin(), links.end() );

            typename std::vector<Link>::iterator pl;
            unsigned int pg;
            pl = links.begin();
            for(pg=0;pg<grid.size();++pg) {
					assert(pl!=links.end());
					grid[pg] = &*pl;
					while( (int)pg == pl->Index() )	{
						++pl;
						if(pl==links.end())
							break;
					}
				}
        }
    };

    void getInBoxTris(const MeshGrid& gm, const BBox& box, std::vector<int>& inBox) {
        // TODO(@seeeagull)
    }

    bool segTriIntersection(const std::pair<vec3f, vec3f>& seg, const std::vector<vec3f>& tri, float& a, float& b) {
        BBox bb0(std::vector<vec3f>{seg.first, seg.second}), bb1(tri);
        vec3f inter;
        if (!bb0.collide(bb1))
            return false;
        else {
            vec3f dir = seg.second - seg.first;
            dir = normalize(dir);

            enum DIRECTION{RIGHT, LEFT, MIDDLE};
            bool inside = true;
            char quadrant[3];
            vec3f candidatePlane;
            for (int i = 0; i < 3; i++) {
                if(seg.first[i] < bb1.min[i]) {
                    quadrant[i] = LEFT;
                    candidatePlane[i] = bb1.min[i];
                    inside = false;
                } else if (seg.first[i] > bb1.max[i]) {
                    quadrant[i] = RIGHT;
                    candidatePlane[i] = bb1.max[i];
                    inside = false;
                } else {
                    quadrant[i] = MIDDLE;
                }
            }

            if(!inside) {
                vec3f maxT;
                for (int i = 0; i < 3; i++) {
                    if (quadrant[i] != MIDDLE && dir[i] !=0.)
                        maxT[i] = (candidatePlane[i]-seg.first[i]) / dir[i];
                    else
                        maxT[i] = -1.;
                }
                int whichPlane = 0;
                for (int i = 1; i < 3; i++)
                    if (maxT[whichPlane] < maxT[i])
                        whichPlane = i;

                if (maxT[whichPlane] < 0.) return false;
            }

            vec3f edge1 = tri[1] - tri[0];
            vec3f edge2 = tri[2] - tri[0];
            vec3f pvec = cross(dir, edge2);
            float det = dot(edge1, pvec);
            vec3f tvec = seg.first - tri[0];
            float inv_det = 1.0 / det;
            vec3f qvec = cross(tvec, edge1);

            bool intersect = true;
            if (det > EPSIL) {
                a = dot(tvec, pvec);
                if ( a < 0.0 ||  a > det)
                intersect = false;
                b = dot(dir, qvec);
                if ( b < 0.0 ||  a +  b > det)
                intersect = false;
            } else if(det < -EPSIL) {
                a = dot(tvec, pvec);
                if ( a > 0.0 ||  a < det)
                intersect = false;
                b = dot(dir, qvec);
                if ( b > 0.0 ||  a +  b < det)
                intersect = false;
            } else 
                intersect = false;

            float orig_dist = dot(edge2, qvec) * inv_det;
            a *= inv_det;
            b *= inv_det;

            if(intersect)
                return (orig_dist>=0 && orig_dist<=distance(seg.second, seg.first));
            return false;
        }
    }

    bool triTriIntersection(const std::vector<vec3f>& p0, const std::vector<vec3f>& p1) {
        vec3f E1 = p0[1] - p0[0];
        vec3f E2 = p0[2] - p0[0];
        vec3f N1 = normalize(cross(E1, E2));
        float d1 = -dot(N1, p0[0]);

        float du0 = dot(N1, p1[0]) + d1;
        float du1 = dot(N1, p1[1]) + d1;
        float du2 = dot(N1, p1[2]) + d1;
        /* coplanarity robustness check */
        #ifdef USE_TRI_TRI_INT_EPSILON_TEST
        if(FABS(du0)<TRI_TRI_INT_EPSILON) du0=0.0;
        if(FABS(du1)<TRI_TRI_INT_EPSILON) du1=0.0;
        if(FABS(du2)<TRI_TRI_INT_EPSILON) du2=0.0;
        #endif
        float du0du1 = du0 * du1;
        float du0du2 = du0 * du2;
        if(du0du1>0.0f && du0du2>0.0f)
            return 0;

        E1 = p1[1] - p1[0];
        E2 = p1[2] - p1[0];
        vec3f N2 = normalize(cross(E1, E2));
        float d2 = -dot(N2, p1[0]);

        float dv0 = dot(N2, p0[0]) + d2;
        float dv1 = dot(N2, p0[1]) + d2;
        float dv2 = dot(N2, p0[2]) + d2;
        #ifdef USE_TRI_TRI_INT_EPSILON_TEST
        if(FABS(dv0)<TRI_TRI_INT_EPSILON) dv0=0.0;
        if(FABS(dv1)<TRI_TRI_INT_EPSILON) dv1=0.0;
        if(FABS(dv2)<TRI_TRI_INT_EPSILON) dv2=0.0;
        #endif
        float dv0dv1 = dv0 * dv1;
        float dv0dv2 = dv0 * dv2;
        if(dv0dv1>0.0f && dv0dv2>0.0f)
            return 0;

        vec3f D = cross(N1, N2);
        short index = 0;
        float max = std::fabs(D[0]);
        float bb = std::fabs(D[1]);
        float cc = std::fabs(D[2]);
        if(bb>max) max=bb,index=1;
        if(cc>max) max=cc,index=2;

        float vp0 = p0[0][index];
        float vp1 = p0[1][index];
        float vp2 = p0[2][index];
        float up0 = p1[0][index];
        float up1 = p1[1][index];
        float up2 = p1[2][index];

        float a,b,c,x0,x1;
        if(dv0dv1>0.0f) {
            a=vp2; b=(vp0-vp2)*dv2; c=(vp1-vp2)*dv2; x0=dv2-dv0; x1=dv2-dv1;
        } else if(dv0dv2>0.0f) {
            a=vp1; b=(vp0-vp1)*dv1; c=(vp2-vp1)*dv1; x0=dv1-dv0; x1=dv1-dv2;
        } else if(dv1*dv2>0.0f || dv0!=0.0f) {
            a=vp0; b=(vp1-vp0)*dv0; c=(vp2-vp0)*dv0; x0=dv0-dv1; x1=dv0-dv2;
        } else if(dv1!=0.0f) {
            a=vp1; b=(vp0-vp1)*dv1; c=(vp2-vp1)*dv1; x0=dv1-dv0; x1=dv1-dv2;
        } else if(dv2!=0.0f) {
            a=vp2; b=(vp0-vp2)*dv2; c=(vp1-vp2)*dv2; x0=dv2-dv0; x1=dv2-dv1;
        } else {
            // @seeeagull: shouldn't reach here
        }

        float d,e,f,y0,y1;
        if(du0du1>0.0f) {
            d=up2; e=(up0-up2)*du2; f=(up1-up2)*du2; y0=du2-du0; y1=du2-du1;
        } else if(du0du2>0.0f) {
            d=up1; e=(up0-up1)*du1; f=(up2-up1)*du1; y0=du1-du0; y1=du1-du2;
        } else if(du1*du2>0.0f || du0!=0.0f) {
            d=up0; e=(up1-up0)*du0; f=(up2-up0)*du0; y0=du0-du1; y1=du0-du2;
        } else if(du1!=0.0f) {
            d=up1; e=(up0-up1)*du1; f=(up2-up1)*du1; y0=du1-du0; y1=du1-du2;
        } else if(du2!=0.0f) {
            d=up2; e=(up0-up2)*du2; f=(up1-up2)*du2; y0=du2-du0; y1=du2-du1;
        } else {
            // @seeeagull: shouldn't reach here
        }

        float xx = x0 * x1;
        float yy = y0 * y1;
        float xxyy = xx * yy;

        float isect1[2], isect2[2];
        float tmp = a * xxyy;
        isect1[0] = tmp + b * x1 * yy;
        isect1[1] = tmp + c * x0 * yy;

        tmp = d * xxyy;
        isect2[0] = tmp + e * xx * y1;
        isect2[1] = tmp + f * xx * y0;

        std::sort(isect1[0],isect1[1]);
        std::sort(isect2[0],isect2[1]);

        if(isect1[1]<isect2[0] || isect2[1]<isect1[0]) return 0;
        return 1;
    }

    bool faceFaceIntersection(const vec3i& f0, const vec3i& f1, const std::vector<vec3f>& p0, const std::vector<vec3f>& p1) {
		int sv = 0;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                if (f0[i] == f1[j])
                        sv++;
		if(sv==3) return true;
		if(sv==0) return (triTriIntersection(p0, p1));
		if(sv==1) {
			int i0, i1;
            bool flag = false;
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++)
                    if (f0[i] == f1[j]) {
                        i0 = i, i1 = j;
                        flag = true;
                        break;
                    }
                if (flag) break;
            }
            float a, b;
			vec3f ps[5] = {p0[i0], p0[(i0+1)%3], p0[(i0+2)%3], p1[(i1+1)%3], p1[(i1+2)%3]};
            for (int i = 0; i < 5; ++i) {
                ps[i] = 0.5f * ps[i];
                if (i > 0)
                    ps[i] = ps[i] + ps[0];
            }
			if(segTriIntersection(std::make_pair(ps[1], ps[2]), p1, a, b)) {
				if(a+b>=1 || a<=EPSIL || b<=EPSIL ) return false;
				return true;
			}
			if(segTriIntersection(std::make_pair(ps[3], ps[4]), p0, a, b)) {
				if(a+b>=1 || a<=EPSIL || b<=EPSIL ) return false;
				return true;
			}

		}
		return false;
    }

    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto &verts = prim->verts;
        auto &faces = prim->tris;
        auto &referred = prim->tris.add_attr<int>("referred", 0);
        auto &intersecting = prim->tris.add_attr<int>("intersecting", 0);

		MeshGrid gM(faces, verts);
        for (int i = 0; i < faces->size(); ++i) {
            auto fi = faces[i];
            referred[i] = 1;
            std::vector<vec3f> pfi{verts[fi[0]], verts[fi[1]], verts[fi[2]]};
            BBox bbox(pfi);
            std::vector<int> inBox{};
            getInBoxTris(gM, bbox, inBox);
            for(auto fib: inBox) {
                std::vector<vec3f> pfib{verts[faces[fib][0]], verts[faces[fib][1]], verts[faces[fib][2]]};
                if(referred[fib] == 0 && (fib != i) )
                    if(faceFaceIntersection(fi, faces[fib], pfi, pfib))
                        intersecting[fi] = intersecting[fib] = 1;
            }
            inBox.clear();
        }
        prim->tris.erase_attr("referred");

        set_output("prim", std::move(prim));
    }
};

ZENO_DEFNODE(SelectSelfIntersectingFaces)
({
    {{"prim"}},
    {("prim")},
    {},
    {"primitive"},
});


} // namespace zeno
