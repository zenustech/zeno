#ifndef __ZENO_DATAMODEL_H__
#define __ZENO_DATAMODEL_H__

#include <QtWidgets>

class NodeDataModel : public QAbstractTableModel
{
    Q_OBJECT
public:
    explicit NodeDataModel(QObject *parent = 0);
    int rowCount(const QModelIndex &parent) const override;
    int columnCount(const QModelIndex &parent) const override;
    QVariant data(const QModelIndex &index, int role) const override;
    QVariant headerData(int section, Qt::Orientation orientation, int role) const override;

private:
    void _initData();

    QStringList m_headers;
    QVector<QVector<float>> m_datas;
};

#endif