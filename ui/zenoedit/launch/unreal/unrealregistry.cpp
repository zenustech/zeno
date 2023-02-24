#include "unrealregistry.h"

void UnrealSubjectRegistry::markDirty(bool flag) {
    m_bIsDirty = flag;
}

bool UnrealSubjectRegistry::isDirty() const {
    return m_bIsDirty;
}
