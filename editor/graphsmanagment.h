#ifndef __GRAPHS_MANAGMENT_H__
#define __GRAPHS_MANAGMENT_H__

class GraphsModel;

#include <QObject>

class GraphsManagment : public QObject
{
    Q_OBJECT
public:
    GraphsManagment(QObject *parent = nullptr);
    GraphsModel *currentModel();
    GraphsModel *openZsgFile(const QString &fn);
    GraphsModel *importGraph(const QString &fn);

private:
    GraphsModel *m_model;
    QString m_currFile;
};

#endif