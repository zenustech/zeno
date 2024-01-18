#include "assetsmodel.h"
#include "graphmodel.h"
#include <zeno/core/Session.h>
#include <zeno/core/Assets.h>


AssetsModel::AssetsModel(QObject* parent)
    : QAbstractListModel(parent)
{
    std::shared_ptr<zeno::Assets> assets =  zeno::getSession().assets;
    m_cbCreateAsset = assets->register_createAsset([&](const std::string& name) {
        _addAsset(QString::fromStdString(name));
    });

    m_cbRemoveAsset = assets->register_removeAsset([&](const std::string& name) {
        _removeAsset(QString::fromStdString(name));
    });

    m_cbRenameAsset = assets->register_renameAsset([&](const std::string& old_name, const std::string& new_name) {
        //TODO
    });
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

int AssetsModel::rowByName(const QString& name) const {
    for (int i = 0; i < m_assets.size(); i++) {
        if (m_assets[i]->name() == name) {
            return i;
        }
    }
    return -1;
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
    zeno::getSession().assets->createAsset(assetName.toStdString());
}

void AssetsModel::addAsset(const zeno::GraphData& graph)
{
}

void AssetsModel::removeAsset(const QString& assetName)
{
    zeno::getSession().assets->removeAsset(assetName.toStdString());
}

void AssetsModel::_addAsset(const QString& newName)
{
    int nRows = m_assets.size();
    beginInsertRows(QModelIndex(), nRows, nRows);

    std::shared_ptr<zeno::Assets> asts = zeno::getSession().assets;
    std::shared_ptr<zeno::Graph> spAsset = asts->getAsset(newName.toStdString());
    auto pNewAsstModel = new GraphModel(spAsset, this);
    m_assets.append(pNewAsstModel);

    endInsertRows();
}

void AssetsModel::_removeAsset(const QString& name)
{
    //this is a private impl method, called by callback function.
    int row = rowByName(name);

    beginRemoveRows(QModelIndex(), row, row);

    GraphModel* pDelete = m_assets[row];
    m_assets.removeAt(row);
    delete pDelete;

    endRemoveRows();
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