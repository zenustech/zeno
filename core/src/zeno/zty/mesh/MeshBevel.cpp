#include <zeno/zty/mesh/MeshBevel.h>


ZENO_NAMESPACE_BEGIN
namespace zty {


void MeshBevel::operator()(Mesh &mesh) const {
    uint32_t base = static_cast<uint32_t>(mesh.vert.size());

    decltype(mesh.vert) new_vert;
    auto new_loop = mesh.loop;

    size_t start = 0;
    for (size_t i = 0; i < mesh.poly.size(); i++) {
        uint32_t npoly = mesh.poly[i];
        [[likely]] if (npoly >= 3) {

            for (uint32_t p = 0; p < npoly; p++) {
                auto last_p = (p - 1 + npoly) % npoly;
                auto m = mesh.loop[start + p];
                auto last_m = mesh.loop[start + last_p];

                new_loop.push_back(base + 3 * p);           // cw
                new_loop.push_back(base + 3 * last_p + 2);  // last_rw
                new_loop.push_back(m);                      // mv
                new_loop.push_back(base + 3 * p + 1);       // mw

                new_loop.push_back(base + 3 * last_p);      // last_cw
                new_loop.push_back(base + 3 * last_p + 1);  // last_mw
                new_loop.push_back(base + 3 * last_p + 2);  // last_rw
                new_loop.push_back(base + 3 * p);           // cw
            }

            for (uint32_t p = 0; p < npoly; p++) {
                auto l = mesh.loop[start + (p - 1 + npoly) % npoly];
                auto m = mesh.loop[start + p];
                auto r = mesh.loop[start + (p + 1) % npoly];

                auto lv = mesh.vert[l];
                auto mv = mesh.vert[m];
                auto rv = mesh.vert[r];

                auto cw = mv + fac * (lv - mv) + fac * (rv - mv);
                auto mw = lerp(mv, rv, fac);
                auto rw = lerp(rv, mv, fac);

                new_vert[m] -= smo * (cw - mv);

                new_vert.push_back(cw);
                new_vert.push_back(mw);
                new_vert.push_back(rw);

                new_loop[start + p] = base + 3 * p;
            }

            base += 3 * npoly;

        }
        start += npoly;
    }

    mesh.vert.insert(mesh.vert.begin(), new_vert.begin(), new_vert.end());
    mesh.poly.resize(mesh.poly.size() + (new_loop.size() - mesh.loop.size()) / 4, 4);
    mesh.loop = std::move(new_loop);
}


}
ZENO_NAMESPACE_END
