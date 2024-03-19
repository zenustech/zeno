#include "zforksubgrapdlg.h"
#include <zeno/utils/logger.h>
#include <zenoui/style/zenostyle.h>
#include "zenoapplication.h"
#include <zenomodel/include/graphsmanagment.h>
#include <zenomodel/include/igraphsmodel.h>
#include <zenomodel/include/nodeparammodel.h>
#include <zenomodel/include/uihelper.h>
#include "variantptr.h"

ZForkSubgraphDlg::ZForkSubgraphDlg(const QMap<QString, QString>& subgs, QWidget* parent)
    : ZFramelessDialog(parent)
    , m_subgsMap(subgs)
{
    initUi();
    QString path = ":/icons/zeno-logo.png";
    this->setTitleIcon(QIcon(path));
    this->setTitleText(tr("Fork Subgraphs"));
    resize(ZenoStyle::dpiScaledSize(QSize(500, 600)));
}

void ZForkSubgraphDlg::initUi()
{
    const QStringList& subgraphs = m_subgsMap.keys();
    QWidget* pWidget = new QWidget(this);
    QVBoxLayout* pLayout = new QVBoxLayout(pWidget);
    m_pTableWidget = new QTableWidget(this);
    m_pTableWidget->verticalHeader()->setVisible(false);
    m_pTableWidget->setProperty("cssClass", "select_subgraph");
    m_pTableWidget->setColumnCount(3);
    QStringList labels = { tr("Subgraph"), tr("Name"), tr("Mtlid") };
    m_pTableWidget->setHorizontalHeaderLabels(labels);
    m_pTableWidget->setShowGrid(false);
    m_pTableWidget->horizontalHeader()->setDefaultAlignment(Qt::AlignLeft);
    m_pTableWidget->setEditTriggers(QAbstractItemView::DoubleClicked);
    m_pTableWidget->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Stretch);
    m_pTableWidget->horizontalHeader()->setSectionResizeMode(2, QHeaderView::Stretch);
    for (const auto& subgraph : subgraphs)
    {
        int row = m_pTableWidget->rowCount();
        m_pTableWidget->insertRow(row);
        QTableWidgetItem* pItem = new QTableWidgetItem(m_subgsMap[subgraph]);
        pItem->setFlags(pItem->flags() & ~Qt::ItemIsEditable);
        m_pTableWidget->setItem(row, 0, pItem);

        QTableWidgetItem* pNameItem = new QTableWidgetItem(subgraph);
        m_pTableWidget->setItem(row, 1, pNameItem);

        QTableWidgetItem* pMatItem = new QTableWidgetItem(subgraph);
        m_pTableWidget->setItem(row, 2, pMatItem);
        pMatItem->setData(Qt::UserRole, subgraph);
        
    }
    pLayout->addWidget(m_pTableWidget); 
    QHBoxLayout* pHLayout = new QHBoxLayout(this);
    QPushButton* pOkBtn = new QPushButton(tr("Ok"), this);
    QPushButton* pCancelBtn = new QPushButton(tr("Cancel"), this);
    m_pImportBtn = new QPushButton(tr("Import Material File"), this);
    int width = ZenoStyle::dpiScaled(80);
    int height = ZenoStyle::dpiScaled(24);
    pOkBtn->setFixedSize(width, height);
    pCancelBtn->setFixedSize(width, height);
    m_pImportBtn->setFixedHeight(height);
    pHLayout->addWidget(m_pImportBtn);
    pHLayout->addStretch();
    pHLayout->addWidget(pOkBtn);
    pHLayout->addWidget(pCancelBtn);
    pLayout->addLayout(pHLayout);
    this->setMainWidget(pWidget);

    connect(pOkBtn, &QPushButton::clicked, this, &ZForkSubgraphDlg::onOkClicked);
    connect(pCancelBtn, &QPushButton::clicked, this, &ZForkSubgraphDlg::reject);
    connect(m_pImportBtn, &QPushButton::clicked, this, &ZForkSubgraphDlg::onImportClicked);
} 

QMap<QString, QMap<QString, QVariant>> ZForkSubgraphDlg::readFile()
{
    QMap<QString, QMap<QString, QVariant>> matValueMap;
    QFile file(m_importPath);
    bool ret = file.open(QIODevice::ReadOnly | QIODevice::Text);
    if (!ret) {
        zeno::log_error("cannot open file: {} ({})", m_importPath.toStdString(),
            file.errorString().toStdString());
        return matValueMap;
    }
    rapidjson::Document doc;
    QByteArray bytes = file.readAll();
    doc.Parse(bytes);

    if (!doc.IsObject())
    {
        zeno::log_error("json file is corrupted");
        return matValueMap;
    }
    auto jsonObject = doc.GetObject();
    QMap<QString, QMap<QString, QString>> matchMap;//<matType, <matKey, paramName>>
    if (jsonObject.HasMember("match"))
    {
        const auto& objVal = jsonObject["match"];
        if (objVal.IsObject())
        {
            for (auto iter = objVal.MemberBegin(); iter != objVal.MemberEnd(); iter++) {
                QString matType = iter->name.GetString();
                if (iter->value.IsObject())
                {
                    for (auto iter1 = iter->value.MemberBegin(); iter1 != iter->value.MemberEnd(); iter1++) {
                        QString paramName = iter1->name.GetString();
                        QString matKey = iter1->value.GetString();
                        matchMap[matType][matKey] = paramName;
                    }
                }
            }
        }
    }
    for (const auto& mtlid : m_subgsMap.keys())
    {
        QString mat = m_subgsMap[mtlid];
        IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();
        auto subgIdx = pGraphsModel->index(mat);
        if (subgIdx.isValid() && !m_matchInfo.contains(mat) && matchMap.isEmpty())
        {
            m_matchInfo[mat].m_matType = mat;
            //preset 
            QModelIndexList nodes = pGraphsModel->searchInSubgraph("SubInput", subgIdx);
            for (const QModelIndex& subInput : nodes)
            {
                NodeParamModel* nodeParams = QVariantPtr<NodeParamModel>::asPtr(subInput.data(ROLE_NODE_PARAMS));
                QModelIndex nameIdx = nodeParams->getParam(PARAM_PARAM, "name");
                QString paramName = nameIdx.data(ROLE_PARAM_VALUE).toString();
                m_matchInfo[mat].m_matchInfo[paramName] = paramName;
            }
        }
        if (jsonObject.HasMember(mtlid.toUtf8()))
        {
            const auto& objVal = jsonObject[mtlid.toStdString().c_str()];
            if (objVal.IsObject())
            {
                for (auto iter = objVal.MemberBegin(); iter != objVal.MemberEnd(); iter++) {
                    QString matKey = iter->name.GetString();
                    QVariant val = UiHelper::parseJson(iter->value);
                    if (matchMap.contains(mat))
                    {
                        const auto& match = matchMap[mat];
                        if (match.contains(matKey))
                        {
                            matKey = match[matKey];
                        }
                    }
                    else if (matchMap.isEmpty())
                    {
                        if (m_matchInfo[mat].m_matchInfo.contains(matKey))
                        {
                            m_matchInfo[mat].m_matchInfo.remove(matKey);
                        }
                        else
                        {
                            m_matchInfo[mat].m_matKeys.insert(matKey);
                        }
                    }
                    
                    matValueMap[mtlid][matKey] = val;
                }
            }
        }
    }

    for (auto iter = m_matchInfo.begin(); iter != m_matchInfo.end();)
    {
        if (iter->m_matchInfo.isEmpty() || iter->m_matKeys.isEmpty())
            m_matchInfo.erase(iter++);
        else
            iter++;
    }

    if (!m_matchInfo.isEmpty())
    {
        QMap<QString, STMatchMatInfo> infos = ZMatchPresetSubgraphDlg::getMatchInfo(m_matchInfo, this);
        if (!infos.isEmpty())
        {
            m_matchInfo = infos;
        }
    }
    return matValueMap;
}

void ZForkSubgraphDlg::onImportClicked()
{
    m_importPath = QFileDialog::getOpenFileName(nullptr, "File to Open", "", "All Files(*);;");
    QStringList strLst = m_importPath.split("/");
    if (!strLst.isEmpty())
        m_pImportBtn->setText(tr("Material File: ") + strLst.last());
}

void ZForkSubgraphDlg::onOkClicked()
{
    IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();
    ZASSERT_EXIT(pGraphsModel);
    int count = m_pTableWidget->rowCount();
    int rowNum = qSqrt(count);
    int colunmNum = count / (rowNum > 0 ? rowNum : 1);
    QMap<QString, QMap<QString, QVariant>> matValueMap;
    if (!m_importPath.isEmpty())
        matValueMap = readFile();
    QPointF pos = m_nodeIndex.data(ROLE_OBJPOS).toPointF();
    const auto& sugIdx = m_nodeIndex.data(ROLE_SUBGRAPH_IDX).toModelIndex();
    for (int row = 0; row < count; row++)
    {
        QString subgName = m_pTableWidget->item(row, 0)->data(Qt::DisplayRole).toString();
        QString name = m_pTableWidget->item(row, 1)->data(Qt::DisplayRole).toString();
        QString mtlid = m_pTableWidget->item(row, 2)->data(Qt::DisplayRole).toString();
        QString old_mtlid = m_pTableWidget->item(row, 2)->data(Qt::UserRole).toString();
        
        const QModelIndex& index = pGraphsModel->forkMaterial(sugIdx, pGraphsModel->index(subgName), name, mtlid, old_mtlid);
        if (!index.isValid())
        {
            QMessageBox::warning(this, tr("warring"), tr("fork preset subgraph '%1' failed.").arg(name));
            continue;
        }
        
        int currC = row / rowNum + 1;
        int currR = row % rowNum;
        QPointF newPos(pos.x() + currC * 600, pos.y() + currR * 600);
        pGraphsModel->ModelSetData(index, newPos, ROLE_OBJPOS);

        if (!matValueMap.contains(old_mtlid))
            continue;
        const auto& valueMap = matValueMap[old_mtlid];
        NodeParamModel* nodeParams = QVariantPtr<NodeParamModel>::asPtr(index.data(ROLE_NODE_PARAMS));
        if (nodeParams)
        {
            for (const auto& inputIdx : nodeParams->getInputIndice())
            {
                QString name = inputIdx.data(ROLE_PARAM_NAME).toString();
                if (m_matchInfo.contains(subgName) && m_matchInfo[subgName].m_matchInfo.contains(name))
                {
                    name = m_matchInfo[subgName].m_matchInfo[name];
                }
                if (valueMap.contains(name))
                {
                    QVariant newVal = valueMap[name];
                    auto deflVal = inputIdx.data(ROLE_PARAM_VALUE);
                    if (deflVal.canConvert<UI_VECTYPE>() && newVal.canConvert<UI_VECTYPE>())
                    {
                        UI_VECTYPE deflVec = deflVal.value<UI_VECTYPE>();
                        UI_VECTYPE newVec = newVal.value<UI_VECTYPE>();
                        newVec.resize(deflVec.size());
                        newVal = QVariant::fromValue(newVec);
                    }
                    if (deflVal.type() != newVal.type())
                    {
                        zeno::log_error("{}-{} value type error", subgName.toStdString(), name.toStdString());
                        continue;
                    }
                    pGraphsModel->ModelSetData(inputIdx, newVal, ROLE_PARAM_VALUE);
                }
            }
        }
    }
    accept();
}

void ZForkSubgraphDlg::setNodeIdex(const QModelIndex& index)
{
    m_nodeIndex = index;
}