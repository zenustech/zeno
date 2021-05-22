#include <vector>
#include <string>
#include <iostream>
#include <cstdlib>
#include <cstring>

#define show(x) (std::cout << #x "=" << (x) << std::endl)

using std::cout;
using std::endl;


template <size_t CHK_SIZE = 512, size_t MAX_ELMS = 64>
class AOSOA
{
    struct ElmInfo {
        size_t size;
        size_t offset;
    };

    size_t m_size;
    ElmInfo m_elms[MAX_ELMS];
    size_t m_elmcount;
    size_t m_stride;

    std::vector<void *> m_chunks;

public:
    explicit AOSOA(std::vector<size_t> const &elmsizes) {
        size_t accum = 0;
        m_elmcount = elmsizes.size();
        for (size_t i = 0; i < m_elmcount; i++) {
            m_elms[i].size = elmsizes[i];
            m_elms[i].offset = accum * CHK_SIZE;
            accum += elmsizes[i];
        }
        m_stride = accum;
    }

    void *address(size_t a, size_t p) const {
        size_t chkidx = p / CHK_SIZE;
        auto elm = m_elms[a];
        size_t chkoff = (p % CHK_SIZE) * elm.size + elm.offset;
        return (void *)((char *)m_chunks[chkidx] + chkoff);
    }

    void resize(size_t n) {
        size_t oldchknr = m_chunks.size();
        size_t chknr = (n + CHK_SIZE - 1) / CHK_SIZE;
        for (size_t i = chknr; i < oldchknr; i++) {
            std::free(m_chunks[i]);
            m_chunks[i] = nullptr;
        }
        m_chunks.resize(chknr);
        for (size_t i = oldchknr; i < chknr; i++) {
            size_t size = CHK_SIZE * m_stride;
            void *p = std::malloc(size);
            m_chunks[i] = p;
        }
        m_size = n;
    }

    size_t size() const {
        return m_size;
    }
};


int main(void)
{
    AOSOA<> a({sizeof(int), sizeof(short), sizeof(char), sizeof(char)});
    a.resize(32);
    a.resize(1000);
    for (int i = 0; i < 1000; i++) {
        *(int *)a.address(0, i) = i + 1;
    }
    for (int i = 0; i < 1000; i++) {
        cout << i << "=" << *(int *)a.address(0, i) << endl;
    }
    a.resize(8);
    a.resize(1024);
    for (int i = 0; i < 14; i++) {
        cout << i << "=" << *(int *)a.address(0, i) << endl;
    }
    show(a.address(0, 0));
    show(a.address(1, 0));
    show(a.address(2, 0));
    show(a.address(3, 0));
    show(a.address(0, 1));
    show(a.address(1, 1));
    show(a.address(2, 1));
    show(a.address(3, 1));
    return 0;
}
