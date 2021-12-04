#include <zeno/zty/mesh/DCEL.h>
#include <zeno/zty/mesh/Mesh.h>
#include <unordered_map>
#include <vector>
#include <tuple>


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
    std::unordered_multimap<Vert const *, Edge const *> vert_leaving;
    for (auto const &e: edge) {
        auto f0 = face_avg.at(e.face);
        auto f1 = face_avg.at(e.twin->face);
        auto e0 = e.origin->co;
        auto e1 = e.twin->origin->co;
        auto ep = (f0 + f1 + e0 + e1) * 0.25f;
        edge_point.emplace(&e, ep);
        vert_leaving.emplace(e.origin, &e);
    }

    Mesh mesh;

    std::unordered_map<Vert const *, math::vec3f> vert_point;
    std::vector<std::tuple<Vert const *, Edge const *, Face const *, Edge const *>> new_faces;
    for (auto const &v: vert) {
        auto [it0, it1] = vert_leaving.equal_range(&v);
        uint32_t n = 0;
        math::vec3f res(0);
        for (auto it = it0; it != it1; ++it) {
            auto e = it->second;
            auto e_prev = e;
            while (e_prev->next != e)
                e_prev = e_prev->next;
            new_faces.emplace_back(&v, e, e->face, e_prev);

            auto ep2 = e->origin->co + e->twin->origin->co;
            auto fp = face_avg.at(e->face);
            res += ep2 + fp;
            n++;
        }

        res += (n - 3) * v.co;
        res *= 1.f / n;
        vert_point.emplace(&v, res);
    }
}


}
ZENO_NAMESPACE_END
