#pragma once

#include <QString>
#include <QList>
#include <QSet>

struct FuzzyMatchKey
{
    QString name;
    QVector<int> matchIndices;
    FuzzyMatchKey() = default;
    FuzzyMatchKey(const QString& name, const QVector<int>& matchIndices)
        : name(name)
        , matchIndices(matchIndices)
    {
    }
};

static QVector<int> key_appear_by_order(const QString& pattern, const QString& key) {
    QVector<int> matchIndices;
    int from = 0;
    for (auto i = 0; i < pattern.size(); i++) {
        QChar c = pattern.at(i);
        auto res = key.indexOf(c, from, Qt::CaseInsensitive);
        if (res == -1) {
            return {};
        }
        matchIndices.push_back(res);
        from = res + 1;
    }
    return matchIndices;
}

static QVector<int> capital_match(const QString& pattern, const QString& key) {
    QVector<int> upperIndices;
    QString only_upper;
    for (auto i = 0; i < key.size(); i++) {
        if (key.at(i).isUpper()) {
            only_upper += key.at(i);
            upperIndices.push_back(i);
        }
    }
    int from = only_upper.indexOf(pattern, 0, Qt::CaseInsensitive);
    if (from == -1) {
        return {};
    }
    QVector<int> matchIndices;
    for (int i = 0; i < pattern.size(); ++i) {
        matchIndices.push_back(upperIndices[from + i]);
    }
    return matchIndices;
}

static void merge_condidates(
    QList<FuzzyMatchKey>& ret_list,
    QSet<QString>& ret_set,
    const QList<FuzzyMatchKey>& lst
) {
    const int MAX_COUNT = 30;
    for (auto i = 0; i < lst.size(); i++) {
        if (ret_list.size() > MAX_COUNT) {
            break;
        }
        auto& s = lst[i].name;
        if (!ret_set.contains(s)) {
            ret_list.push_back(lst[i]);
            ret_set.insert(s);
        }
    }
}

static QList<FuzzyMatchKey> fuzzy_search(const QString& pattern, const QList<QString>& keys) {
    QList<FuzzyMatchKey> key_appear_by_order_conds;
    for (auto i = 0; i < keys.size(); i++) {
        auto& k = keys[i];
        auto indices = key_appear_by_order(pattern, k);
        if (!indices.empty()) {
            key_appear_by_order_conds.push_back({ k, indices });
        }
    }
    QList<FuzzyMatchKey> direct_match_conds;
    for (auto i = 0; i < key_appear_by_order_conds.size(); i++) {
        auto& k = key_appear_by_order_conds[i].name;
        if (k.contains(pattern, Qt::CaseInsensitive)) {
            direct_match_conds.push_back(key_appear_by_order_conds[i]);
        }
    }
    QList<FuzzyMatchKey> prefix_match_conds;
    for (auto i = 0; i < direct_match_conds.size(); i++) {
        auto& k = direct_match_conds[i].name;
        if (k.indexOf(pattern, 0, Qt::CaseInsensitive) == 0) {
            prefix_match_conds.push_back(direct_match_conds[i]);
        }
    }
    QList<FuzzyMatchKey> capital_match_conds;
    for (auto i = 0; i < key_appear_by_order_conds.size(); i++) {
        auto& k = key_appear_by_order_conds[i].name;
        auto indices = capital_match(pattern, k);
        if (!indices.empty()) {
            capital_match_conds.push_back(key_appear_by_order_conds[i]);
        }
    }
    QList<FuzzyMatchKey> ret_list;
    QSet<QString> ret_set;
    merge_condidates(ret_list, ret_set, prefix_match_conds);
    merge_condidates(ret_list, ret_set, capital_match_conds);
    merge_condidates(ret_list, ret_set, direct_match_conds);
    merge_condidates(ret_list, ret_set, key_appear_by_order_conds);
    return ret_list;
}
