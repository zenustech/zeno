#pragma once
#include "zensim/geometry/AdaptiveGrid.hpp"
#include "zensim/geometry/AnalyticLevelSet.h"
#include "zensim/geometry/Collider.h"
#include "zensim/geometry/SparseGrid.hpp"
#include "zensim/resource/Resource.h"
#include <zeno/types/UserData.h>
#include <zeno/zeno.h>

namespace zeno {

struct ZenoAdaptiveGrid : IObjectClone<ZenoAdaptiveGrid> {
    using ag_t = zs::AdaptiveGrid<3, zs::f32, zs::index_sequence<3, 4, 5>>;

    auto &getAdaptiveGrid() noexcept {
        return ag;
    }
    const auto &getAdaptiveGrid() const noexcept {
        return ag;
    }

    template <typename T>
    decltype(auto) setMeta(const std::string &tag, T &&val) {
        return metas[tag] = FWD(val);
    }
    template <typename T = float>
    decltype(auto) readMeta(const std::string &tag, zs::wrapt<T> = {}) const {
        return std::any_cast<T>(metas.at(tag));
    }
    template <typename T = float>
    decltype(auto) readMeta(const std::string &tag, zs::wrapt<T> = {}) {
        return std::any_cast<T>(metas.at(tag));
    }
    bool hasMeta(const std::string &tag) const {
        if (auto it = metas.find(tag); it != metas.end())
            return true;
        return false;
    }
    /// @note -1 implies not a double buffer; 0/1 indicates the current double buffer index.
    int checkDoubleBuffer(const std::string &attr) const {
        auto metaTag = attr + "_cur";
        if (hasMeta(metaTag))
            return readMeta<int>(metaTag);
        return -1;
    }
    bool isDoubleBufferAttrib(const std::string &attr) const {
        if (attr.back() == '0' || attr.back() == '1')
            return true;
        else if (hasMeta(attr + "_cur"))
            return true;
        return false;
    }

    ag_t ag;
    std::map<std::string, std::any> metas;
};

} // namespace zeno