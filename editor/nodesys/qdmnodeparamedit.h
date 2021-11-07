#ifndef QDMNODEPARAMEDIT_H
#define QDMNODEPARAMEDIT_H

#include <zeno/common.h>
#include <QWidget>
#include <vector>

ZENO_NAMESPACE_BEGIN

class QDMNodeParamEdit : public QWidget
{
    Q_OBJECT

public:
    explicit QDMNodeParamEdit(QWidget *parent = nullptr);
    ~QDMNodeParamEdit();

signals:
    void entryModified(QString name);
};

ZENO_NAMESPACE_END

#endif // QDMNODEPARAMEDIT_H
