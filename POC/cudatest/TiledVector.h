#pragma once

#error "WIP"

namespace fdb {

template <class ...Ts>
struct TiledVector {
    struct Tile {
        std::tuple<Ts[8]...> m_data{};
    };

    Vector<Tile> m_arr;
};

}
