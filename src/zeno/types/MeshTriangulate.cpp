#include <zeno/types/MeshTriangulate.h>
#include <zeno/zycl/parallel_scan.h>


ZENO_NAMESPACE_BEGIN
namespace types {


#ifndef ZENO_WITH_ZYCL
zycl::vector<math::vec3f> meshToTriangleVertices(Mesh const &mesh) {
    decltype(auto) vert = mesh.vert.to_vector();
    decltype(auto) loop = mesh.loop.to_vector();
    decltype(auto) poly = mesh.poly.to_vector();

    zycl::vector<math::vec3f> ret;
    {
        decltype(auto) tris = ret.as_vector();
        tris.reserve(poly.size() * 3);
        for (auto const &[p_start, p_num]: poly) {
            if (p_num <= 2) continue;
            int first = loop[p_start];
            int last = loop[p_start + 1];
            for (int l = p_start + 2; l < p_start + p_num; l++) {
                int now = loop[l];
                tris.push_back(vert[first]);
                tris.push_back(vert[last]);
                tris.push_back(vert[now]);
                last = now;
            }
        }
    }
    return ret;
}
#else
zycl::vector<math::vec3f> meshToTriangleVertices(Mesh const &mesh) {
    zycl::vector<int> indices(mesh.poly.size());

    zycl::default_queue().submit([&] (zycl::handler &cgh) {
        auto axr_indices = zycl::make_access<zycl::wd>(cgh, indices);
        auto axr_poly = zycl::make_access<zycl::ro>(cgh, mesh.poly);

        cgh.parallel_for
        ( zycl::range<1>(indices.size())
        , [=] (zycl::item<1> it) {
            auto const &[p_start, p_num] = axr_poly[it[0]];
            axr_indices[it[0]] = p_num < 2 ? 0 : p_num - 2;
        });
    });

    auto sum = zycl::parallel_scan<256>(indices, indices.size());

    zycl::vector<math::vec3f> tris;
    {
        auto axr_sum = zycl::host_access<zycl::ro>(sum);
        tris.resize(axr_sum[0] * 3);
    }

    zycl::default_queue().submit([&] (zycl::handler &cgh) {
        auto axr_indices = zycl::make_access<zycl::ro>(cgh, indices);
        auto axr_poly = zycl::make_access<zycl::ro>(cgh, mesh.poly);
        auto axr_loop = zycl::make_access<zycl::ro>(cgh, mesh.loop);
        auto axr_vert = zycl::make_access<zycl::ro>(cgh, mesh.vert);
        auto axr_tris = zycl::make_access<zycl::wd>(cgh, tris);

        cgh.parallel_for
        ( zycl::range<1>(indices.size())
        , [=] (zycl::item<1> it) {
            auto const &[p_start, p_num] = axr_poly[it[0]];
            if (p_num < 2) return;
            auto base = axr_indices[it[0]];

            int first = axr_loop[p_start];
            int last = axr_loop[p_start + 1];
            for (int i = 0; i < p_num - 2; i++) {
                int now = axr_loop[p_start + 2 + i];
                int ind = (base + i) * 3;
                axr_tris[ind + 0] = axr_vert[first];
                axr_tris[ind + 1] = axr_vert[last];
                axr_tris[ind + 2] = axr_vert[now];
                last = now;
            }
        });
    });

    return tris;
}
#endif


}
ZENO_NAMESPACE_END
