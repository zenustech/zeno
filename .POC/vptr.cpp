#include <bits/stdc++.h>

using std::cout;
using std::endl;

template <class T>
struct vector_abstract {
    std::vector<char> m_data;
    size_t m_stride = 0;
    size_t m_count = 0;

    template <class S>
    emplace_back() {
        m_stride = sizeof(S);
        m_data.resize(m_data.size() + m_stride);
        S();
    }
};

struct IObject {
    virtual show() const = 0;
};

struct ManObject : IObject {
    virtual show() const {
        printf("man!\n");
    }
};

int main()
{
    vector_abstract<IObject> os;

    os.emplace_back<ManObject>();

    return 0;
}
