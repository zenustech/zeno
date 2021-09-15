namespace fdb {


static constexpr size_t BAD_OFFSET = (size_t)-1;


template <class T, size_t Dim, size_t N0, size_t N1>
struct L1PointerMap {
    Vector<T> m_data;
    Vector<size_t> m_offset1;

    L1PointerMap()
        : m_offset1(1 << (Dim * N1), BAD_OFFSET)
    {}

    template <auto Mode = Access::read_write, class Handler>
    auto accessor(Handler hand) {
        auto dataAxr = m_data.template accessor<Mode>(hand);
        auto offset1Axr = m_offset1.template accessor<Access::read>(hand);
        return [=] (vec<Dim, size_t> indices) -> T * {
            auto offset1 = *offset1Axr(indices >> N0);
            if (offset1 == BAD_OFFSET)
                return nullptr;
            offset1 *= 1 << (Dim * N0);
            size_t offset0 = indices & ((1 << N0) - 1);
            return dataAxr(offset1 | offset0);
        };
    }

    static inline constexpr auto size() {
        return vec<Dim, size_t>(1 << (N0 + N1));
    }
};


}
