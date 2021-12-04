#include <zeno/zty/mesh/DCEL.h>
#include <unordered_map>


ZENO_NAMESPACE_BEGIN
namespace zty {


void DCEL::subdivision()
{
    std::unordered_map<Face const *, math::vec3f> face_avg;
    for (auto const &f: face) {
        auto e = f.first;
        auto f_avg = e->origin->co;
        size_t f_num = 1;
        e = e->next;
        while (e != f.first) {
            f_avg += e->origin->co;
            e = e->next;
            f_num++;
        }
        f_avg *= 1.f / f_num;
        face_avg.emplace(&f, f_avg);
    }

    std::unordered_map<Edge const *, math::vec3f> edge_point;
    for (auto const &e: edge) {
        auto f0 = face_avg.at(e.face);
        auto f1 = face_avg.at(e.twin->face);
        auto e0 = e.origin->co;
        auto e1 = e.twin->origin->co;
        auto ep = (f0 + f1 + e0 + e1) * 0.25f;
        edge_point.emplace(&e, ep);
    }
}


}
ZENO_NAMESPACE_END
