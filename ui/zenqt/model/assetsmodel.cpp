#include "assetsmodel.h"
#include "graphmodel.h"
#include <zeno/core/Session.h>
#include <zeno/core/Assets.h>
#include <zenoio/writer/zdawriter.h>
#include <zeno/utils/log.h>


AssetsModel::AssetsModel(QObject* parent)
    : QAbstractListModel(parent)
{
    std::shared_ptr<zeno::AssetsMgr> assets =  zeno::getSession().assets;
    m_cbCreateAsset = assets->register_createAsset([&](zeno::AssetInfo info) {
        _addAsset(info);
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

GraphModel* AssetsModel::getAssetGraph(const QString& graphName) const
{
    for (int i = 0; i < m_assets.length(); i++) {
        if (m_assets[i].pGraphM->name() == graphName) {
            return m_assets[i].pGraphM;
        }
    }
    return nullptr;
}

zeno::AssetInfo AssetsModel::getAsset(const QString& assetName) const
{
    for (int i = 0; i < m_assets.length(); i++) {
        if (m_assets[i].info.name == assetName.toStdString()) {
            return m_assets[i].info;
        }
    }
    return zeno::AssetInfo();
}

int AssetsModel::rowByName(const QString& name) const {
    for (int i = 0; i < m_assets.size(); i++) {
        if (m_assets[i].pGraphM->name() == name) {
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
    int row = index.row();
    if (row >= 0 && row < m_assets.size()) {
        if (Qt::DisplayRole == role || ROLE_CLASS_NAME == role) {
            GraphModel* pAsset = m_assets[row].pGraphM;
            return pAsset->name();
        }
    }
    //todo
    return QVariant();
}

bool AssetsModel::setData(const QModelIndex& index, const QVariant& value, int role)
{
    return false;
}

void AssetsModel::newAsset(const zeno::AssetInfo info)
{
    auto& assets = zeno::getSession().assets;
    assets->createAsset(info);

    zeno::Asset asset = assets->getAsset(info.name);

    zeno::ZenoAsset zasset;
    zasset.info = info;
    zasset.graph = asset.sharedGraph->exportGraph();
    zasset.inputs = asset.inputs;
    zasset.outputs = asset.outputs;

    zenoio::ZdaWriter writer;
    const std::string& content = writer.dumpAsset(zasset);

    QString filePath = QString::fromStdString(info.path);
    QFile f(filePath);
    if (!f.open(QIODevice::WriteOnly)) {
        qWarning() << Q_FUNC_INFO << "Failed to open" << filePath << f.errorString();
        zeno::log_error("Failed to open file for write: {} ({})", filePath.toStdString(),
            f.errorString().toStdString());
        return;
    }

    f.write(content.c_str());
    f.close();
    zeno::log_info("saved '{}' successfully", filePath.toStdString());
}

void AssetsModel::addAsset(const zeno::GraphData& graph)
{
}

void AssetsModel::removeAsset(const QString& assetName)
{
    zeno::getSession().assets->removeAsset(assetName.toStdString());
}

void AssetsModel::_addAsset(zeno::AssetInfo info)
{
    int nRows = m_assets.size();
    beginInsertRows(QModelIndex(), nRows, nRows);

    std::shared_ptr<zeno::AssetsMgr> asts = zeno::getSession().assets;
    std::shared_ptr<zeno::Graph> spAsset = asts->getAsset(info.name).sharedGraph;
    auto pNewAsstModel = new GraphModel(spAsset, nullptr, this);

    _AssetItem item;
    item.info = info;
    item.pGraphM = pNewAsstModel;
    m_assets.append(item);

    endInsertRows();
}

void AssetsModel::_removeAsset(const QString& name)
{
    //this is a private impl method, called by callback function.
    int row = rowByName(name);

    beginRemoveRows(QModelIndex(), row, row);

    GraphModel* pDelete = m_assets[row].pGraphM;
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