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
    void reloadGraph(const QString& graphName);
    bool saveCurrent();
    void clear();
    void removeCurrent();

private:
    GraphsModel *m_model;
    QString m_currFile;
};

#endif