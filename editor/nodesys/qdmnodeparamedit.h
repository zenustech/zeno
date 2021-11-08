#ifndef QDMNODEPARAMEDIT_H
#define QDMNODEPARAMEDIT_H

#include <zeno/common.h>
#include "qdmgraphicsnode.h"
#include <QFormLayout>
#include <QWidget>

ZENO_NAMESPACE_BEGIN

class QDMNodeParamEdit : public QWidget
{
    Q_OBJECT

    QFormLayout *layout;
    QDMGraphicsNode *currNode{};

public:
    explicit QDMNodeParamEdit(QWidget *parent = nullptr);
    ~QDMNodeParamEdit();

public slots:
    void setCurrentNode(QDMGraphicsNode *node);

signals:
    void entryModified(QString name);
};

ZENO_NAMESPACE_END

#endif // QDMNODEPARAMEDIT_H
