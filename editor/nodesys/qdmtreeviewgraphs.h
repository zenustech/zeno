#ifndef QDMTREEVIEWGRAPHS_H
#define QDMTREEVIEWGRAPHS_H

#include <zeno/common.h>
#include "qdmgraphicsscene.h"
#include <QTreeView>
#include <QStandardItem>
#include <QStandardItemModel>
#include <memory>
#include <vector>

ZENO_NAMESPACE_BEGIN

class QDMTreeViewGraphs : public QTreeView
{
    Q_OBJECT

    QDMGraphicsScene *rootScene;
    QStandardItemModel *model;

    std::vector<std::unique_ptr<QStandardItem>> raiiItems;

public:
    explicit QDMTreeViewGraphs(QWidget *parent = nullptr);
    ~QDMTreeViewGraphs();

    void setRootScene(QDMGraphicsScene *scene);

signals:
    void entryClicked(QString name);
};

ZENO_NAMESPACE_END

#endif // QDMTREEVIEWGRAPHS_H
