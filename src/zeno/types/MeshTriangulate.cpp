#include <zeno/types/MeshTriangulate.h>
#include <zeno/zycl/parallel_scan.h>


ZENO_NAMESPACE_BEGIN
namespace types {


void meshToTrianglesCPU(Mesh const &mesh, std::vector<math::vec3f> &vertices) {
    decltype(auto) vert = mesh.vert.to_vector();
    decltype(auto) loop = mesh.loop.to_vector();
    decltype(auto) poly = mesh.poly.to_vector();

    vertices.clear();
    vertices.reserve(poly.size() * 3);
    for (auto const &[p_start, p_num]: poly) {
        if (p_num <= 2) continue;
        int first = loop[p_start];
        int last = loop[p_start + 1];
        for (int l = p_start + 2; l < p_start + p_num; l++) {
            int now = loop[l];
            vertices.push_back(vert[first]);
            vertices.push_back(vert[last]);
            vertices.push_back(vert[now]);
            last = now;
        }
    }
}


void meshToTriangles(Mesh const &mesh, zycl::vector<std::array<math::vec3f, 3>> &tris) {
    zycl::vector<int> indices(mesh.poly.size());

    zycl::default_queue().submit([&] (zycl::handler &cgh) {
        auto axr_indices = zycl::make_access<zycl::wd>(cgh, indices);
        auto axr_poly = zycl::make_access<zycl::ro>(cgh, mesh.poly);

        cgh.parallel_for
        ( zycl::range<1>(indices.size())
        , [=] (zycl::item<1> it) {
            auto const &[p_start, p_num] = axr_poly[it[0]];
            axr_indices[it[0]] = p_num;
        });
    });

    zycl::parallel_scan<256>(indices, indices.size());

    zycl::default_queue().submit([&] (zycl::handler &cgh) {
        auto axr_indices = zycl::make_access<zycl::ro>(cgh, indices);
        auto axr_poly = zycl::make_access<zycl::ro>(cgh, mesh.poly);
        auto axr_loop = zycl::make_access<zycl::ro>(cgh, mesh.loop);
        auto axr_vert = zycl::make_access<zycl::ro>(cgh, mesh.vert);
        auto axr_tris = zycl::make_access<zycl::ro>(cgh, tris);

        cgh.parallel_for
        ( zycl::range<1>(indices.size())
        , [=] (zycl::item<1> it) {
            auto const &[p_start, p_num] = mesh.poly[it[0]];
            auto base = axr_indices[it[0]];

            int first = axr_loop[p_start];
            int last = axr_loop[p_start + 1];
            for (int i = 0; i < p_num - 2; i++) {
                int now = axr_loop[p_start + 2 + i];
                tris[base + i] = axr_vert[first];
                last = now;
            }
        });
    });
}


}
ZENO_NAMESPACE_END
