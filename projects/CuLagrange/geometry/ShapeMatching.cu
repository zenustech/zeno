#include "Structures.hpp"
#include "zensim/Logger.hpp"
// #include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/io/MeshIO.hpp"
#include "zensim/math/bit/Bits.h"
#include "zensim/types/Property.h"
#include <atomic>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>

// #include "../geometry/kernel/tiled_vector_ops.hpp"
// #include "../geometry/kernel/topology.hpp"
// #include "../geometry/kernel/geo_math.hpp"

#include "zensim/math/Rotation.hpp"
#include "zensim/math/matrix/QRSVD.hpp"


namespace zeno {

struct MatchTransformation : zeno::INode {

    virtual void apply() override {
        using namespace zs;
        using vec3 = zs::vec<float,3>;
        using vec4 = zs::vec<float,4>;
        using mat3 = zs::vec<float,3,3>;
        
        constexpr auto space = execspace_e::openmp;
        auto ompPol = omp_exec();
        constexpr auto exec_tag = wrapv<space>{};

        auto rprim = get_input<zeno::PrimitiveObject>("refObj");
        auto tprim = get_input<zeno::PrimitiveObject>("targetObj");

        auto shape_size = rprim->verts.size();

        // compute the center of mass of rprim
        auto rcm = vec3::zeros();
        auto tcm = vec3::zeros();
        // cmVec[0] = zs::vec<float,3>::zeros();
        
        const auto& rverts = rprim->verts;
        const auto& tverts = tprim->verts;

        ompPol(zs::range(shape_size),[
            exec_tag = exec_tag,
            &rverts,&tverts,&rcm,&tcm] (int vi) mutable {
                for(int d = 0;d != 3;++d) {
                    atomic_add(exec_tag,&rcm[d],rverts[vi][d]);
                    atomic_add(exec_tag,&tcm[d],tverts[vi][d]);
                }
        });

        rcm /= shape_size;
        tcm /= shape_size;

        std::vector<mat3> dAs(shape_size,mat3::zeros());
        ompPol(zs::range(shape_size),[
            &dAs,rcm,tcm,&rverts,&tverts] (int vi) mutable {
                auto q = vec3::from_array(rverts[vi]) - rcm;
                auto p = vec3::from_array(tverts[vi]) - tcm;
                dAs[vi] = dyadic_prod(p,q);
        });

        auto A = mat3::zeros();
        ompPol(zs::range(shape_size * 9),[
            exec_tag = exec_tag,
            &A,
            &dAs] (int dof) mutable {
                auto dAid = dof / 9;
                auto Aoffset = dof % 9;
                auto r = Aoffset / 3;
                auto c = Aoffset % 3;
                const auto& dA = dAs[dAid];
                atomic_add(exec_tag,&A[r][c],dA[r][c]);                
        });
        A /= shape_size;

        auto [R,S] = math::polar_decomposition(A);

        // R = R.transpose();

        printf("R:\n%f\b%f\b%f\n%f\b%f\b%f\n%f\b%f\b%f\n",
            (float)R(0,0),(float)R(0,1),(float)R(0,2),
            (float)R(1,0),(float)R(1,1),(float)R(1,2),
            (float)R(2,0),(float)R(2,1),(float)R(2,2));

        auto b = tcm - R * rcm;
        
        auto q = vec4::zeros();
        auto m00 = R(0,0);
        auto m01 = R(0,1);
        auto m02 = R(0,2);
        auto m10 = R(1,0);
        auto m11 = R(1,1);
        auto m12 = R(1,2);
        auto m20 = R(2,0);
        auto m21 = R(2,1);
        auto m22 = R(2,2); 
        // float t{0};

        // if (m22 < 0) {
        //     if (m00 > m11) {
        //         t = 1 + m00 -m11 -m22;
        //         q = vec4( t, m01+m10, m20+m02, m12-m21 );
        //     }
        //     else {
        //         t = 1 -m00 + m11 -m22;
        //         q = vec4( m01+m10, t, m12+m21, m20-m02 );
        //     }
        // }
        // else {
        //     if (m00 < -m11) {
        //         t = 1 -m00 -m11 + m22;
        //         q = vec4( m20+m02, m12+m21, t, m01-m10 );
        //     }
        //     else {
        //         t = 1 + m00 + m11 + m22;
        //         q = vec4( m12-m21, m20-m02, m01-m10, t );
        //     }
        // }
        // q *= 0.5 / zs::sqrt(t);

        auto trace = m00 + m11 + m22;
        if (trace > 0.0f)
        {
            auto k = 0.5f / zs::sqrt(1.0f + trace);
            q = vec4( k * (m12 - m21), k * (m20 - m02), k * (m01 - m10), 0.25f / k );
        }
        else if ((m00 > m11) && (m00 > m22))
        {
            auto k = 0.5f / zs::sqrt(1.0f + m00 - m11 - m22);
            q = vec4( 0.25f / k, k * (m10 + m01), k * (m20 + m02), k * (m12 - m21) );
        }
        else if (m11 > m22)
        {
            auto k = 0.5f / zs::sqrt(1.0f + m11 - m00 - m22);
            q = vec4( k * (m10 + m01), 0.25f / k, k * (m21 + m12), k * (m20 - m02) );
        }
        else
        {
            auto k = 0.5f / zs::sqrt(1.0f + m22 - m00 - m11);
            q = vec4( k * (m20 + m02), k * (m21 + m12), 0.25f / k, k * (m01 - m10) );
        }

        // due to the column-major setting, need a quaternion negation here
        q[0] = -q[0];
        q[1] = -q[1];
        q[2] = -q[2];
        
        auto retq = std::make_shared<zeno::NumericObject>();
        retq->set<zeno::vec4f>(zeno::vec4f(q[0],q[1],q[2],q[3]));

        auto retb = std::make_shared<zeno::NumericObject>();
        retb->set<zeno::vec3f>(zeno::vec3f(b[0],b[1],b[2]));


        set_output("quat",std::move(retq));
        set_output("trans",std::move(retb));
    }

};

ZENDEFNODE(MatchTransformation,{
    {{"refObj"},{"targetObj"}},
    {"quat","trans"},
    {},
    {"Geometry"},
});

};