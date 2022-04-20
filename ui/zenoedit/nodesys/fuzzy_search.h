#pragma once

#include <QString>
#include <QList>
#include <QSet>

static bool key_appear_by_order(const QString& pattern, QString key) {
	for (auto i = 0; i < pattern.size(); i++) {
		QChar c = pattern.at(i);
		auto res = key.indexOf(c, 0, Qt::CaseInsensitive);
		if (res == -1) {
			return false;
		}
		key = key.mid(res + 1);
	}
	return true;
}

static bool capital_match(const QString& pattern, const QString& key) {
	QString only_upper;
	for (auto i = 0; i < key.size(); i++) {
		if (key.at(i).isUpper()) {
			only_upper += key.at(i);
		}
	}
	return only_upper.contains(pattern, Qt::CaseInsensitive);
}

static void merge_condidates(
	QList<QString>& ret_list,
	QSet<QString>& ret_set,
	const QList<QString>& lst
) {
	const int MAX_COUNT = 30;
	for (auto i = 0; i < lst.size(); i++) {
		if (ret_list.size() > MAX_COUNT) {
			break;
		}
		auto s = lst[i];
		if (!ret_set.contains(s)) {
			ret_list.push_back(s);
			ret_set.insert(s);
		}
	}
}

static QList<QString> fuzzy_search(const QString& pattern, const QList<QString>& keys) {
	QList<QString> key_appear_by_order_conds;
	for (auto i = 0; i < keys.size(); i++) {
		auto k = keys[i];
		if (key_appear_by_order(pattern, k)) {
			key_appear_by_order_conds.push_back(k);
		}
	}
	QList<QString> direct_match_conds;
	for (auto i = 0; i < key_appear_by_order_conds.size(); i++) {
		auto k = key_appear_by_order_conds[i];
		if (k.contains(pattern, Qt::CaseInsensitive)) {
			direct_match_conds.push_back(k);
		}
	}
	QList<QString> prefix_match_conds;
	for (auto i = 0; i < direct_match_conds.size(); i++) {
		auto k = direct_match_conds[i];
		if (k.indexOf(pattern, 0, Qt::CaseInsensitive) == 0) {
			prefix_match_conds.push_back(k);
		}
	}
	QList<QString> capital_match_conds;
	for (auto i = 0; i < key_appear_by_order_conds.size(); i++) {
		auto k = key_appear_by_order_conds[i];
		if (capital_match(pattern, k)) {
			capital_match_conds.push_back(k);
		}
	}
	QList<QString> ret_list;
	QSet<QString> ret_set;
	merge_condidates(ret_list, ret_set, prefix_match_conds);
	merge_condidates(ret_list, ret_set, capital_match_conds);
	merge_condidates(ret_list, ret_set, direct_match_conds);
	merge_condidates(ret_list, ret_set, key_appear_by_order_conds);
	return ret_list;
}
