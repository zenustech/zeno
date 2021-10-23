#ifndef QDMMOUSEEVENTEATER_H
#define QDMMOUSEEVENTEATER_H

#include <QObject>

class QDMMouseEventEater : public QObject
{
    Q_OBJECT
public:
    explicit QDMMouseEventEater(QObject *parent = nullptr);

    virtual bool eventFilter(QObject *object, QEvent *event) override;
};

#endif // QDMMOUSEEVENTEATER_H
