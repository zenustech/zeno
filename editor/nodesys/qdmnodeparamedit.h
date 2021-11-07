#ifndef QDMNODEPARAMEDIT_H
#define QDMNODEPARAMEDIT_H

#include <zeno/common.h>
#include <QListView>
#include <QStandardItemModel>
#include <QStandardItem>
#include <vector>

ZENO_NAMESPACE_BEGIN

class QDMNodeParamEdit : public QListView
{
    Q_OBJECT

    std::unique_ptr<QStandardItemModel> model;
    std::vector<std::unique_ptr<QStandardItem>> items;

public:
    explicit QDMNodeParamEdit(QWidget *parent = nullptr);
    ~QDMNodeParamEdit();

signals:
    void entryModified(QString name);
};

ZENO_NAMESPACE_END

#endif // QDMNODEPARAMEDIT_H
