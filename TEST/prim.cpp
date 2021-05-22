#include <vector>
#include <string>


struct AOSOA
{
    static constexpr size_t CHK_SIZE = 512;

    size_t m_count;
    std::vector<size_t> m_offsets;
    size_t m_stride;

    struct ChunkPtr {
        char *m_ptr;

        ChunkPtr() {
            m_ptr = new char[CHK_SIZE];
        }

        ~ChunkPtr() {
            delete m_ptr;
            m_ptr = nullptr;
        }

        void *get() const {
            return (void *)m_ptr;
        }
    };

    std::vector<ChunkPtr> m_chunks;

    void *address(size_t a, size_t p) const {
        size_t chkoff = m_stride * (p % CHK_SIZE) + m_offsets[a];
        size_t chkidx = p / CHK_SIZE;
        return (void *)((char *)m_chunks[chkidx].get() + chkoff);
    }

    void resize(size_t n) {
        size_t chknr = (n + CHK_SIZE - 1) / CHK_SIZE;
        m_chunks.resize(chknr);
        m_count = n;
    }

    size_t size() const {
        return m_count;
    }
};


int main(void)
{
    AOSOA a;
}
