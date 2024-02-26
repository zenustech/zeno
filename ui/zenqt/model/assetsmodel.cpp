#include "assetsmodel.h"
#include "graphmodel.h"
#include <zeno/core/Session.h>
#include <zeno/core/Assets.h>
#include <zenoio/writer/zdawriter.h>
#include <zeno/utils/log.h>
#include <zenoio/reader/zdareader.h>


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

    _initAssets();
}

AssetsModel::~AssetsModel()
{

}

void AssetsModel::_initAssets()
{
    QDir dir(QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation));
    dir.mkpath("Zeno/assets");
    dir.cd("Zeno");
    dir.cd("assets");
    for (const QFileInfo& file : dir.entryInfoList(QDir::Files))
    {
        QString path = file.filePath();
        _initAsset(path);
    }
}

void AssetsModel::_initAsset(const QString& path)
{
    zenoio::ZdaReader reader;
    zenoio::ZSG_PARSE_RESULT result = reader.openFile(path.toStdString());
    if (!result.bSucceed) {
        return;
    }

    zeno::ZenoAsset zasset = reader.getParsedAsset();
    zasset.info.path = path.toStdString();
    auto& assets = zeno::getSession().assets;
    assets->createAsset(zasset);
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
    zeno::ZenoAsset asset;
    asset.info = info;

    zeno::GraphData& sample = asset.graph;
    sample.type = zeno::Subnet_Main;
    sample.name = info.name;

    zeno::NodeData input1;
    input1.name = "input1";
    input1.cls = "SubInput";
    input1.uipos = { 0, 0 };

    zeno::NodeData input2;
    input2.name = "input2";
    input2.cls = "SubInput";
    input2.uipos = { 0,700 };

    zeno::NodeData output1;
    output1.name = "output1";
    output1.cls = "SubOutput";
    output1.uipos = { 1300, 250 };

    zeno::NodeData output2;
    output2.name = "output2";
    output2.cls = "SubOutput";
    output2.uipos = { 1300, 900 };

    sample.nodes.insert(std::make_pair("input1", input1));
    sample.nodes.insert(std::make_pair("input2", input2));
    sample.nodes.insert(std::make_pair("output1", output1));
    sample.nodes.insert(std::make_pair("output2", output2));

    std::vector<zeno::ParamInfo>& inputs = asset.inputs;
    std::vector<zeno::ParamInfo>& outputs = asset.outputs;

    zeno::ParamInfo param;
    param.name = "input1";
    param.bInput = true;
    param.socketType = zeno::PrimarySocket;
    inputs.push_back(param);

    param.name = "input2";
    param.bInput = true;
    param.socketType = zeno::PrimarySocket;
    inputs.push_back(param);

    param.name = "output1";
    param.bInput = false;
    param.socketType = zeno::PrimarySocket;
    outputs.push_back(param);

    param.name = "output2";
    param.bInput = false;
    param.socketType = zeno::PrimarySocket;
    outputs.push_back(param);

    assets->createAsset(asset);
    saveAsset(QString::fromStdString(info.name));
}

void AssetsModel::addAsset(const zeno::GraphData& graph)
{
}

void AssetsModel::removeAsset(const QString& assetName)
{
    zeno::getSession().assets->removeAsset(assetName.toStdString());
}

void AssetsModel::saveAsset(const QString& name)
{
    auto& assets = zeno::getSession().assets;
    zeno::Asset asset = assets->getAsset(name.toStdString());

    zeno::ZenoAsset zasset;
    zasset.info = asset.m_info;
    zasset.graph = asset.sharedGraph->exportGraph();
    zasset.inputs = asset.inputs;
    zasset.outputs = asset.outputs;

    zenoio::ZdaWriter writer;
    const std::string& content = writer.dumpAsset(zasset);

    QString filePath = QString::fromStdString(zasset.info.path);
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