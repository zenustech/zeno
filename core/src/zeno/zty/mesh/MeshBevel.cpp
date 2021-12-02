#include <zeno/zty/mesh/MeshBevel.h>


ZENO_NAMESPACE_BEGIN
namespace zty {


void MeshBevel::operator()(Mesh &mesh) const {
    uint32_t base = static_cast<uint32_t>(mesh.vert.size());

    size_t start = 0;
    size_t poly_count = mesh.poly.size();
    for (size_t i = 0; i < poly_count; i++) {
        uint32_t npoly = mesh.poly[i];
        [[likely]] if (npoly >= 3) {

            for (uint32_t p = 0; p < npoly; p++) {
                auto last_p = (p - 1 + npoly) % npoly;
                auto m = mesh.loop[start + p];
                auto last_m = mesh.loop[start + last_p];

                mesh.loop.push_back(base + 3 * p);           // cw
                mesh.loop.push_back(base + 3 * p + 1);       // mw
                mesh.loop.push_back(base + 3 * last_p + 2);  // last_rw
                mesh.loop.push_back(m);                      // mv

                mesh.loop.push_back(base + 3 * last_p);      // last_cw
                mesh.loop.push_back(base + 3 * last_p + 1);  // last_mw
                mesh.loop.push_back(base + 3 * last_p + 2);  // last_rw
                mesh.loop.push_back(base + 3 * p);           // cw

                mesh.poly.push_back(4);
                mesh.poly.push_back(4);
            }

            for (uint32_t p = 0; p < npoly; p++) {
                auto l = mesh.loop[start + (p - 1 + npoly) % npoly];
                auto &m = mesh.loop[start + p];
                auto r = mesh.loop[start + (p + 1) % npoly];
                auto &lv = mesh.vert[l];
                auto &mv = mesh.vert[m];
                auto &rv = mesh.vert[r];
                auto cw = mv + fac * (lv - mv) + fac * (rv - mv);
                auto mw = lerp(mv, rv, fac);
                auto rw = lerp(rv, mv, fac);
                mesh.vert.push_back(cw);
                mesh.vert.push_back(mw);
                mesh.vert.push_back(rw);
                base += 3;

                m = base + 3 * p;
            }

        }
        start += npoly;
    }
}


}
ZENO_NAMESPACE_END
