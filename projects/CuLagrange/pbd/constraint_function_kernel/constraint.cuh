#pragma once

namespace zeno { namespace CONSTRAINT {

    // template<typename VECTOR3d,typename SCALER>
    // constexpr bool edge_length_constraint_predicate(const VECTOR3d x[2],
    //     const SCALER& rl)
    // {
    //     auto edge = x[0] - x[1];
    //     auto en = edge.norm();
    //     return en > rl;
    // }

    template<typename VECTOR3d,typename SCALER>
    constexpr void edge_length_constraint_projection(const VECTOR3d x[2],
        const SCALER& rl,
        const SCALER inv_m[2],
        const SCALER& compliance,
        VECTOR3d dx[2]) {
            auto edge = x[0] - x[1];
            auto en = edge.norm();
            edge /= (en + (SCALER)1e-6);
            auto C = en - rl;
            auto w = inv_m[0] + inv_m[1];
            auto s = -C / (w + compliance);

            dx[0] = edge * s * inv_m[0];
            dx[1] = -edge * s * inv_m[1];
    }

    template<typename VECTOR3d,typename SCALER>
    constexpr void tet_volume_constraint_projection(const VECTOR3d x[4],
        const SCALER& rv,
        const SCALER inv_m[4],
        const SCALER& compliance,
        VECTOR3d dx[4]) {
            dx[0] = (x[1] - x[2]).cross(x[3] - x[2]);
            dx[1] = (x[2] - x[0]).cross(x[3] - x[0]);
            dx[2] = (x[0] - x[1]).cross(x[3] - x[1]);
            dx[3] = (x[1] - x[0]).cross(x[2] - x[0]);

            SCALER w = 0;
            for(int i = 0;i != 4;++i)
                w += dx[i].l2NormSqr() * inv_m[i];
            SCALER vol = zs::abs(x[1] - x[0]).cross(x[2] - x[0]).dot(x[3] - x[0]) / (SCALER)6;
            SCALER C = (vol - rv) * (SCALER)6;
            SCALER s = -C / (w + compliance);

            for(int i = 0;i != 4;++i)
                dx[i] = dx[i] * s * inv_m[i];
    }

    // edge0 = [0,2,1]
    // edge1 = [0,1,3]
    template<typename VECTOR3d,typename SCALER>
    constexpr void tri_bending_constraint_projection(const VECTOR3d x[4],
        const SCALER& rda,
        const SCALER inv_m[4],
        const SCALER& compliance,
        VECTOR3d dx[4]) {
            VECTOR3d p[4];
            for(int i = 0;i != 4;++i)
                p[i] = x[i] - x[0];
            auto p12 = p[1].cross(p[2]);
            auto p13 = p[1].cross(p[3]);
            auto n0 = p12 / (p23.norm() + (SCALER)1e-6);
            auto n1 = p13 / (p12.norm() + (SCALER)1e-6);

            auto d = n1.dot(n0);
            auto p12n = p12.norm();
            auto p13n = p13.norm();

            VECTOR3d q[4];

            auto q[2] = (p[1].cross(n1) + n0.cross(p[1]) * d)/p12n;
            auto q[3] = (p[1].cross(n0) + n1.cross(p[1]) * d)/p13n;
            auto q[1] = -(p[2].cross(n1) + n0.cross(p[2]) *d)/p12n - (p[3].cross(n0) + n1.cross(p[3]) * d)/p13n;
            auto q[0] = -q[1] - q[2] - q[3];

            auto C = zs::acos(d) - rda;
            SCALER w = 0;
            for(int i = 0;i != 4;++i)
                w += q[i].l2NormSqr() * inv_m[i];
            auto s = -C * zs::sqrt(1 - d * d) / (w + compliance);
            for(int i = 0;i != 4;++i)
                dx[i] = q[i] * s * inv_m[i];
    }

};
};