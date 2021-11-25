#ifndef QDMNODEPARAMEDIT_H
#define QDMNODEPARAMEDIT_H

#include <zeno/common.h>
#include "qdmgraphicsnode.h"
#include <QFormLayout>
#include <QWidget>
#include <functional>
#include <string>

ZENO_NAMESPACE_BEGIN

class QDMNodeParamEdit final : public QWidget
{
    Q_OBJECT

    QFormLayout *const layout;
    QDMGraphicsNode *currNode{};

    void invalidateNode(QDMGraphicsNode *node) const;

public:
    explicit QDMNodeParamEdit(QWidget *parent = nullptr);
    ~QDMNodeParamEdit() override;

    QWidget *makeEditForType(QDMGraphicsNode *node, int sockid,
                             std::string const &type);
    void addRow(QString const &name, QWidget *row);
    void clearRows();

public slots:
    void setCurrentNode(QDMGraphicsNode *node);
};

ZENO_NAMESPACE_END

#endif // QDMNODEPARAMEDIT_H
