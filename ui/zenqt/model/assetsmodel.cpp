#include "assetsmodel.h"
#include "graphmodel.h"


AssetsModel::AssetsModel(QObject* parent)
    : QAbstractListModel(parent)
{

}

AssetsModel::~AssetsModel()
{

}

void AssetsModel::init(const zeno::AssetsData& assets)
{
    //todo
}

void AssetsModel::clear()
{

}

GraphModel* AssetsModel::getAsset(const QString& graphName) const
{
    for (int i = 0; i < m_assets.length(); i++) {
        if (m_assets[i]->name() == graphName) {
            return m_assets[i];
        }
    }
    return nullptr;
}

int AssetsModel::rowCount(const QModelIndex& parent) const
{
    return m_assets.size();
}

QVariant AssetsModel::data(const QModelIndex& index, int role) const
{
    GraphModel* pAsset = m_assets[index.row()];
    //todo
    return QVariant();
}

bool AssetsModel::setData(const QModelIndex& index, const QVariant& value, int role)
{
    return false;
}

void AssetsModel::newAsset(const QString& assetName)
{
    //todo
}

void AssetsModel::addAsset(const zeno::GraphData& graph)
{
    //todo
}

void AssetsModel::removeAsset(const QString& assetName)
{
    //todo
}

QModelIndexList AssetsModel::match(const QModelIndex& start, int role,
    const QVariant& value, int hits,
    Qt::MatchFlags flags) const
{
    //todo
    return QModelIndexList();
}

bool AssetsModel::removeRows(int row, int count, const QModelIndex& parent)
{
    return false;
}

QHash<int, QByteArray> AssetsModel::roleNames() const
{
    return QHash<int, QByteArray>();
}