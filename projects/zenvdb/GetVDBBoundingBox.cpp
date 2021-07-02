#include <zeno/zeno.h>
#include <zeno/VDBGrid.h>
#include <zeno/NumericObject.h>
#include <zeno/PrimitiveObject.h>
#include <openvdb/tree/TreeIterator.h>

namespace zeno {


template <class...>
struct vlist {
};

template <class T, class ...Ts>
struct vlist<T, Ts...> {
    using head = T;
    using rest = vlist<Ts...>;
};

template <template <class, template <class> class> class Tp,
    class L, template <class> class Op>
struct vlistfor {
    template <class ...Ts>
    bool operator()(Ts &&...ts) {
        if (Tp<typename L::head, Op>()(
                std::forward<Ts>(ts)...)) {
            return true;
        } else {
            return vlistfor<Tp, typename L::rest, Op>()(
                std::forward<Ts>(ts)...);
        }
    }
};

template <template <class, template <class> class> class Tp,
    template <class> class Op>
struct vlistfor<Tp, vlist<>, Op> {
    template <class ...Ts>
    bool operator()(Ts &&...ts) {
        return false;
    }
};

template <class D, template <class> class Op>
struct dyncast_derived {
    template <class B, class ...Ts>
    bool operator()(B *bp, Ts &&...ts) {
        auto dp = dynamic_cast<D *>(bp);
        if (!dp)
            return false;
        Op<D>()(dp, std::forward<Ts>(ts)...);
        return true;
    }
};

using vlist_of_vdb_type = vlist<VDBFloatGrid, VDBFloat3Grid,
    VDBIntGrid, VDBInt3Grid>;


template <class GridT>
void calcVdbBounds(GridT *grid, vec3f &bmin, vec3f &bmax)
{
    for (auto leaf = grid->tree().cbeginLeaf(); leaf; ++leaf) {
        auto o = leaf->origin();
        auto p = grid->indexToWorld(o);
        printf("%f %f %f\n", p[0], p[1], p[2]);
        //for (auto iter = leaf->cbeginValueOn(); iter; ++iter) {
        //}
    }
}

template <class D>
struct opCalcVdbBounds {
    void operator()(D *grid, vec3f &bmin, vec3f &bmax) {
        calcVdbBounds(grid->m_grid.get(), bmin, bmax);
    };
};

struct GetVDBBoundingBox : INode {
    virtual void apply() override {
        auto ggrid = get_input<VDBGrid>("vdbGrid");
        zeno::vec3f bmin, bmax;
        vlistfor<dyncast_derived, vlist_of_vdb_type, opCalcVdbBounds>()(
            ggrid.get(), bmin, bmax);

        auto boundMin = std::make_shared<NumericObject>();
        boundMin->set(bmin);
        auto boundMax = std::make_shared<NumericObject>();
        boundMax->set(bmax);
        set_output("boundMin", std::move(boundMin));
        set_output("boundMax", std::move(boundMax));
    }
};

ZENDEFNODE(GetVDBBoundingBox, {
    {"vdbGrid"},
    {"boundMin", "boundMax"},
    {},
    {"openvdb"},
});



}
