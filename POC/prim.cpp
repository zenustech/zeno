#include <vector>
#include <string>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <chrono>

#define show(x) (std::cout << #x "=" << (x) << std::endl)

using std::cout;
using std::endl;


template <size_t CHK_SIZE = 4096, size_t MAX_ELMS = 64>
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
    size_t n = 64 * 1024 * 1024;

    // ordered access AOSOA
    {
        AOSOA<> a({sizeof(float), sizeof(float), sizeof(float)});
        a.resize(n);

        auto t0 = std::chrono::steady_clock::now();

        for (int i = 0; i < n; i++) {
            auto &x = *(float *)a.address(0, i);
            auto &y = *(float *)a.address(1, i);
            auto &z = *(float *)a.address(2, i);
            x = (float)drand48();
            y = (float)drand48();
            z = (float)drand48();
        }

        float ret = 0.0;
        for (int i = 0; i < n; i++) {
            auto &x = *(float *)a.address(0, i);
            auto &y = *(float *)a.address(1, i);
            auto &z = *(float *)a.address(2, i);
            ret += x + y + z;
        }
        cout << ret << endl;

        auto t1 = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        cout << ms << " ms" << endl;
    }

    // ordered access c-array
    {
        float *a = new float[n * 3];

        auto t0 = std::chrono::steady_clock::now();

        for (int i = 0; i < n; i++) {
            auto &x = a[i + n * 0];
            auto &y = a[i + n * 1];
            auto &z = a[i + n * 2];
            x = (float)drand48();
            y = (float)drand48();
            z = (float)drand48();
        }

        float ret = 0.0;
        for (int i = 0; i < n; i++) {
            auto &x = a[i + n * 0];
            auto &y = a[i + n * 1];
            auto &z = a[i + n * 2];
            ret += x + y + z;
        }
        cout << ret << endl;

        auto t1 = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        cout << ms << " ms" << endl;
    }

    // random access AOSOA
    {
        AOSOA<> a({sizeof(float), sizeof(float), sizeof(float)});
        a.resize(n);

        auto t0 = std::chrono::steady_clock::now();

        for (int i = 0; i < n; i++) {
            auto &x = *(float *)a.address(0, i);
            auto &y = *(float *)a.address(1, i);
            auto &z = *(float *)a.address(2, i);
            x = (float)drand48();
            y = (float)drand48();
            z = (float)drand48();
        }

        float ret = 0.0;
        for (int i = 0; i < n; i++) {
            int j = (i * 12345 + 1234567) % n;
            auto &x = *(float *)a.address(0, j);
            auto &y = *(float *)a.address(1, j);
            auto &z = *(float *)a.address(2, j);
            ret += x + y + z;
        }
        cout << ret << endl;

        auto t1 = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        cout << ms << " ms" << endl;
    }

    // random access c-array
    {
        float *a = new float[n * 3];

        auto t0 = std::chrono::steady_clock::now();

        for (int i = 0; i < n; i++) {
            auto &x = a[i + n * 0];
            auto &y = a[i + n * 1];
            auto &z = a[i + n * 2];
            x = (float)drand48();
            y = (float)drand48();
            z = (float)drand48();
        }

        float ret = 0.0;
        for (int i = 0; i < n; i++) {
            int j = (i * 12345 + 1234567) % n;
            auto &x = a[i + n * 0];
            auto &y = a[i + n * 1];
            auto &z = a[i + n * 2];
            ret += x + y + z;
        }
        cout << ret << endl;

        auto t1 = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        cout << ms << " ms" << endl;
    }

    return 0;
}
