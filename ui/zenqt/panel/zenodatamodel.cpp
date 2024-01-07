#include "zenodatamodel.h"


NodeDataModel::NodeDataModel(QObject* parent)
    : QAbstractTableModel(parent)
{
    _initData();
}

int NodeDataModel::rowCount(const QModelIndex& parent) const
{
    return m_datas.size();
}

int NodeDataModel::columnCount(const QModelIndex &parent) const
{
    return m_datas[0].size();
}

QVariant NodeDataModel::data(const QModelIndex& index, int role) const
{
    if (!index.isValid())
        return QVariant();

    if (Qt::TextAlignmentRole == role)
    {
        return int(Qt::AlignLeft | Qt::AlignVCenter);
    }
    else if (Qt::DisplayRole == role)
    {
        return m_datas[index.row()][index.column()];
    }
    return QVariant();
}

QVariant NodeDataModel::headerData(int section, Qt::Orientation orientation, int role) const
{
    if (Qt::DisplayRole != role)
        return QVariant();

    if (orientation == Qt::Horizontal)
    {
        return m_headers[section];
    }
    else if (orientation == Qt::Vertical)
    {
        return section;
    }
    return QVariant();
}

void NodeDataModel::_initData()
{
    m_headers = QStringList({"P[x]", "P[y]", "P[z]", "id", "pscale", "v[x]", "v[y]", "v[z]", "group:particles", "group:stream_p"});
    m_datas = {{-5.717, 115.062, -1.38975, -1, 0.072, 0.0, 0.0, 0.0, 0, 0},
               {-5.806, 113.01, -1.38975, -1, 0.072, 0.0, 0.0, 0.0, 0, 0},
               {-5.66158, 114.03, -1.38975, -1, 0.072, 0.0, 0.0, 0.0, 0, 0},
               {-5.763, 114.562, -1.38975, -1, 0.072, 0.0, 0.0, 0.0, 0, 0},
               {-5.85, 115.23, -1.38975, -1, 0.072, 0.0, 0.0, 0.0, 0, 0},
               {-5.775, 114.21, -1.38975, -1, 0.072, 0.0, 0.0, 0.0, 0, 0},
               {-5.932, 117.234, -1.38975, -1, 0.072, 0.0, 0.0, 0.0, 0, 0},
               {-5.7, 121.43, -1.38975, -1, 0.072, 0.0, 0.0, 0.0, 0, 0},
               {-5.84, 115.062, -1.38975, -1, 0.072, 0.0, 0.0, 0.0, 0, 0},
               {-6.2, 115.062, -1.38975, -1, 0.072, 0.0, 0.0, 0.0, 0, 0},
               {-6.1, 115.062, -1.38975, -1, 0.072, 0.0, 0.0, 0.0, 0, 0},
               {-6.032, 115.062, -1.38975, -1, 0.072, 0.0, 0.0, 0.0, 0, 0},
               {-5.915, 115.062, -1.38975, -1, 0.072, 0.0, 0.0, 0.0, 0, 0},
               {-5.88, 115.062, -1.38975, -1, 0.072, 0.0, 0.0, 0.0, 0, 0},
               {-5.214, 115.062, -1.38975, -1, 0.072, 0.0, 0.0, 0.0, 0, 0}
            };
}