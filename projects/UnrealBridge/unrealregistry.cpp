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

std::vector<UnrealSessionInfo> UnrealSessionRegistry::all() {
    std::vector<UnrealSessionInfo> tmp;
    for (const auto& kv : m_session_info) {
        if (kv.second.udp_address.has_value() && kv.second.udp_port.has_value()) {
            tmp.push_back(kv.second);
        }
    }
    return tmp;
}

void UnrealSessionRegistry::removeSession(const std::string &sessionName) {
    m_session_info.erase(sessionName);
}

EZenoSubjectApplyResult ZenoSubjectRegistry::apply(const std::string &subjectName, const EZenoSubjectType inSubjectType,
                                                   zeno::INode &context) {
    if (auto iterSubject = subjects.find(subjectName); iterSubject != subjects.end() && iterSubject->second) {
        if (const EZenoSubjectType subjectType = iterSubject->second->type(); subjectType != EZenoSubjectType::Invalid && subjectType < EZenoSubjectType::End) {
            if (subjectType != inSubjectType) {
                return EZenoSubjectApplyResult::TypeNotMatch;
            }
            if (auto iterDispatcher = zeno::IUnrealZenoSubject::SubjectApplyDispatchMap.find(subjectType); iterDispatcher != zeno::IUnrealZenoSubject::SubjectApplyDispatchMap.end()) {
                iterDispatcher->second->apply(iterSubject->second, context);
                return EZenoSubjectApplyResult::Success;
            } else {
                return EZenoSubjectApplyResult::UnregisterType;
            }
        } else {
            return EZenoSubjectApplyResult::InvalidType;
        }
    }
    return EZenoSubjectApplyResult::NotFound;
}

void ZenoSubjectRegistry::put(const std::string &subjectName, const std::shared_ptr<zeno::IUnrealZenoSubject> &subject) {
    subjects.insert_or_assign(subjectName, subject);
}
