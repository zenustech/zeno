#pragma once

#include <vector>
#include <set>
#include <memory>


template <class T>
void removeElemsByIndice(std::vector<T>& vec, std::set<int> sorted_indice) {
    if (sorted_indice.empty())
        return;

    for (auto iter = sorted_indice.rbegin(); iter != sorted_indice.rend(); iter++) {
        int rmIdx = *iter;
        vec.erase(vec.begin() + rmIdx);
    }
}

template<class T>
void removeElements(std::vector<std::shared_ptr<T>>& vec, const std::set<int>& indice) {
    if (indice.empty())
        return;

    std::set<T*> sPtrs;
    for (auto idx : indice) {
        sPtrs.insert(vec[idx].get());
    }
    vec.erase(std::remove_if(vec.begin(), vec.end(), [&](const auto& val) {
        auto ptr = val.get();
        auto iter = sPtrs.find(ptr);
        if (iter != sPtrs.end()) {
            sPtrs.erase(iter);
            return true;
        }
        else {
            return false;
        }
    }), vec.end());
}