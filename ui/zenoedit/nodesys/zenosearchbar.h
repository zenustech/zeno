#ifndef __ZENO_SEARCHBAR_H__
#define __ZENO_SEARCHBAR_H__

#include <zenoui/nodesys/nodesys_common.h>
#include <zenoui/include/igraphsmodel.h>

class SubGraphModel;

class ZenoSearchBar : public QWidget
{
	Q_OBJECT

public:
    ZenoSearchBar(const QModelIndex& idx, QWidget *parentWidget = nullptr);

signals:
    void searchRequest(const QString&, SEARCH_RANGE, SEARLCH_ELEMENT, int);
    void searchReached(SEARCH_RECORD rec);

private slots:
    void onSearchForward();
    void onSearchBackward();
    void onSearchExec(const QString& content);

private:
    SEARCH_RECORD _getRecord();

    SEARCH_RANGE m_range;
    SEARLCH_ELEMENT m_elem;
    QString m_content;
    QVector<SEARCH_RECORD> m_records;
    QModelIndexList m_results;
    QLineEdit* m_pLineEdit;
    SubGraphModel* m_model;
    QPersistentModelIndex m_index;
    int m_idx;
};


#endif