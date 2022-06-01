#include "skinning_header.h"

#include <igl/readOBJ.h>
#include <igl/readTGF.h>
#include <igl/readMESH.h>


namespace{
using namespace zeno;
struct MarkSurfacePoints : zeno::INode {
    static constexpr auto s_pi = 3.1415926535897932384626433832795028841972L;
    static constexpr auto s_half_pi = 1.5707963267948966192313216916397514420986L;
    virtual void apply() override {
        // we only support 3d simplex volumetric meshing
        auto shape = get_input<zeno::PrimitiveObject>("shape");
        auto &stag = shape->add_attr<float>("surface_tag");
        shape->resize(shape->size());

        if(shape->quads.size() == 0){
            // if it is not a volume mesh, then we mark all the vertices as surface vertices
            std::fill(stag.begin(),stag.end(),1.0);
            set_output("res",shape);
            return;
        }
        // if the input shape is a volumetric mesh, mark the surface vertices by calculating the 
        // solid angle of the checked vertex
        // the buffer store the solid angles
        std::vector<float> srs;
        srs.resize(shape->size(),0);

        for(size_t i = 0;i < shape->quads.size();++i){
            auto tet = shape->quads[i];
            for(size_t k = 0;k < 4;++k){
                size_t l = (k+1) % 4;
                size_t m = (k+2) % 4;
                size_t n = (k+3) % 4;

                auto v0 = shape->verts[tet[k]];
                auto v1 = shape->verts[tet[l]];
                auto v2 = shape->verts[tet[m]];
                auto v3 = shape->verts[tet[n]];

                auto v10 = v1 - v0;
                auto v20 = v2 - v0;
                auto v30 = v3 - v0;

                auto l10 = zeno::length(v10);
                auto l20 = zeno::length(v20);
                auto l30 = zeno::length(v30);

                auto alpha = zeno::acos(zeno::dot(v10,v20)/l10/l20);
                auto beta = zeno::acos(zeno::dot(v10,v30)/l10/l30);
                auto gamma = zeno::acos(zeno::dot(v20,v30)/l20/l30);

                auto s = 0.5 * (alpha + beta + gamma);

                auto omega = 4*zeno::atan(zeno::sqrt(zeno::tan(s/2)*zeno::tan((s - alpha)/2)*zeno::tan((s-beta)/2)*zeno::tan((s-gamma)/2)));

                srs[tet[k]] += omega;
            }
        }
        // for interior points, the surrounded solid angles should sum up to 4*pi
        // std::fill(stag.begin(),stag.end(),0.0);
        // size_t nm_surface_verts = 0;
        for(size_t i = 0;i < shape->size();++i){
            // std::cout << "STAG<" << i << ">\t: " << srs[i] << "\t" << 4*s_pi << std::endl;
            if(zeno::abs(srs[i] - 4*s_pi) > 1e-3){
                stag[i] = 1.0;
                // nm_surface_verts++;
            }else{
                stag[i] = 0.0;
            }
            // fiberDir[i] = zeno::vec3f(0.0,1.0,0.0);
        }
        // std::cout << "NM_SURFACE_VERTS : " << nm_surface_verts << std::endl;

        set_output("res",shape);
    }
};

ZENDEFNODE(MarkSurfacePoints, {
    {"shape"},
    {"res"},
    {},
    {"Skinning"},
});

};