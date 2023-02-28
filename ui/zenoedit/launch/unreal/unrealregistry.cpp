#include "unrealregistry.h"

void UnrealSubjectRegistry::markDirty(bool flag) {
    m_bIsDirty = flag;
}

bool UnrealSubjectRegistry::isDirty() const {
    return m_bIsDirty;
}
std::string UnrealSessionRegistry::newSession() {
    std::string name = newSessionName();
    UnrealSessionInfo info {};
    m_session_info.insert(std::make_pair(name, std::move(info)));
    return name;

}
bool UnrealSessionRegistry::updateSession(const std::string &sessionName, const UnrealSessionInfo &info) {
    if (m_session_info.find(sessionName) == m_session_info.end()) return false;

    m_session_info.insert_or_assign(sessionName, info);

    if (info.udp_address == "0.0.0.0") {
        m_session_info.at(sessionName).udp_address = "127.0.0.1";
    }

}

const UnrealSessionInfo& UnrealSessionRegistry::getSessionInfo(const std::string &sessionName) {
    return m_session_info.at(sessionName);
}

bool UnrealSessionRegistry::hasSession(const std::string &sessionName) {
    return m_session_info.find(sessionName) != m_session_info.end();
}

std::vector<UnrealSessionInfo*> UnrealSessionRegistry::all() {
    std::vector<UnrealSessionInfo *> tmp;
    for (auto kv : m_session_info) {
        tmp.push_back(&kv.second);
    }
    return tmp;
}
