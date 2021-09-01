#include <zeno/utils/UserData.h>

namespace zeno {

ZENO_API zany &UserData::at(std::string const &name) {
    auto it = m_data.find(name);
    if (it == m_data.end()) {
        return m_data[name];
    }
    return it->second;
}

}
