#ifndef QDMTREEVIEWGRAPHS_H
#define QDMTREEVIEWGRAPHS_H

#include <zeno/common.h>
#include <QTreeView>
#include <QStandardItem>
#include <memory>
#include <vector>

ZENO_NAMESPACE_BEGIN

class QDMTreeViewGraphs : public QTreeView
{
    Q_OBJECT

    std::vector<std::unique_ptr<QStandardItem>> items;

public:
    explicit QDMTreeViewGraphs(QWidget *parent = nullptr);
    ~QDMTreeViewGraphs();

signals:
    void entryClicked(QString name);
};

ZENO_NAMESPACE_END

#endif // QDMTREEVIEWGRAPHS_H
