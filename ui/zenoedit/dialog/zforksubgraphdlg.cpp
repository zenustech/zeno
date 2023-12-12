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

void ZForkSubgraphDlg::onImportClicked()
{
    m_importPath = QFileDialog::getOpenFileName(nullptr, "File to Open", "", "All Files(*);;");
    QStringList strLst = m_importPath.split("/");
    if (!strLst.isEmpty())
        m_pImportBtn->setText(tr("Material File: ") + strLst.last());
}

void ZForkSubgraphDlg::onOkClicked()
{
    int count = m_pTableWidget->rowCount();
    int rowNum = qSqrt(count);
    int colunmNum = count / (rowNum > 0 ? rowNum : 1);
    QPointF pos;
    for (int row = 0; row < count; row++)
    {
        QString subgName = m_pTableWidget->item(row, 0)->data(Qt::DisplayRole).toString();
        QString name = m_pTableWidget->item(row, 1)->data(Qt::DisplayRole).toString();
        QString mtlid = m_pTableWidget->item(row, 2)->data(Qt::DisplayRole).toString();
        QString old_mtlid = m_pTableWidget->item(row, 2)->data(Qt::UserRole).toString();
        IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();
        const QModelIndex& index = pGraphsModel->forkMaterial(pGraphsModel->index(subgName), name, mtlid, old_mtlid);
        if (!index.isValid())
        {
            QMessageBox::warning(this, tr("warring"), tr("fork preset subgraph '%1' failed.").arg(name));
            continue;
        }
        if (row > 0)
        {
            int currC = row / rowNum;
            int currR = row % rowNum;
            QPointF newPos(pos.x() + currC * 600, pos.y() + currR * 600);
            pGraphsModel->ModelSetData(index, newPos, ROLE_OBJPOS);
        }
        else
        {
            pos = index.data(ROLE_OBJPOS).toPointF();
        }

        QFile file(m_importPath);
        bool ret = file.open(QIODevice::ReadOnly | QIODevice::Text);
        if (!ret) {
            zeno::log_error("cannot open file: {} ({})", m_importPath.toStdString(),
                file.errorString().toStdString());
            return;
        }

        rapidjson::Document doc;
        QByteArray bytes = file.readAll();
        doc.Parse(bytes);

        if (!doc.IsObject())
        {
            zeno::log_error("json file is corrupted");
            return;
        }
        auto jsonObject = doc.GetObject();
        
        if (jsonObject.HasMember(mtlid.toUtf8()))
        {
            const auto& objVal = jsonObject[mtlid.toStdString().c_str()];
            NodeParamModel* nodeParams = QVariantPtr<NodeParamModel>::asPtr(index.data(ROLE_NODE_PARAMS));
            if (nodeParams)
            {
                for (const auto& inputIdx : nodeParams->getInputIndice())
                {
                    QString name = inputIdx.data(ROLE_PARAM_NAME).toString();
                    if (objVal.HasMember(name.toUtf8()))
                    {
                        QVariant val = UiHelper::parseJson(objVal[name.toStdString().c_str()]);
                        pGraphsModel->ModelSetData(inputIdx, val, ROLE_PARAM_VALUE);
                    }
                }
            }
        }
    }
    accept();
}