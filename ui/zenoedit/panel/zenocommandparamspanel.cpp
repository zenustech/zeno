#include "zenocommandparamspanel.h"
#include "zenoapplication.h"
#include <zenomodel/include/graphsmanagment.h>
#include <zenoui/comctrl/zwidgetfactory.h>
#include <zenomodel/include/jsonhelper.h>
#include <zenomodel/include/uihelper.h>
#include "nodesview/zenographseditor.h"
#include "zenomainwindow.h"
#include <zenoui/style/zenostyle.h>

ZenoCommandParamsPanel::ZenoCommandParamsPanel(QWidget* parent)
    : QWidget(parent)
{
    initUi();
    initConnection();
}

void ZenoCommandParamsPanel::keyPressEvent(QKeyEvent* event)
{
    if (event->key() == Qt::Key_Delete)
    {
        if (const QItemSelectionModel* pSelectionModel = m_pTableWidget->selectionModel())
        {
            const QModelIndexList& lst = pSelectionModel->selectedRows();
            for (const auto& index : lst)
            {
                auto graphsMgm = zenoApp->graphsManagment();
                ZASSERT_EXIT(graphsMgm);
                IGraphsModel* pModel = graphsMgm->currentModel();
                ZASSERT_EXIT(pModel);
                const QString& path = index.siblingAtColumn(0).data(Qt::UserRole).toString();
                pModel->removeCommandParam(path);
            }
        }
    }
    QWidget::keyPressEvent(event);
}

void ZenoCommandParamsPanel::initUi()
{
    m_pTableWidget = new QTableWidget(this);
    m_pTableWidget->verticalHeader()->setVisible(false);
    m_pTableWidget->horizontalHeader()->setVisible(false);
    m_pTableWidget->setColumnCount(2);
    m_pTableWidget->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    m_pTableWidget->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Stretch);
    m_pTableWidget->setProperty("cssClass", "commandParams");
    m_pTableWidget->setShowGrid(false);
    QVBoxLayout* pLayout = new QVBoxLayout(this);
    m_pExportButton = new QPushButton(tr("Save Params"), this);
    m_pExportButton->setProperty("cssClass", "commandButton");
    m_pExportButton->setFixedWidth(ZenoStyle::dpiScaled(120));
    QHBoxLayout* pHLayout = new QHBoxLayout;
    pHLayout->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    pHLayout->addWidget(m_pExportButton);
    pLayout->addLayout(pHLayout);
    pLayout->addWidget(m_pTableWidget);
    pLayout->setMargin(0);
}

void ZenoCommandParamsPanel::initConnection()
{
    connect(m_pTableWidget, &QTableWidget::itemClicked, this, &ZenoCommandParamsPanel::onItemClicked);
    connect(m_pTableWidget, &QTableWidget::itemChanged, this, &ZenoCommandParamsPanel::onItemChanged);
    connect(m_pExportButton, &QPushButton::clicked, this, &ZenoCommandParamsPanel::onExport);
    connect(zenoApp->graphsManagment(), &GraphsManagment::modelInited, this, &ZenoCommandParamsPanel::onModelInited);
    onModelInited();
}

void ZenoCommandParamsPanel::appendRow(const QString& path, const QString& name)
{
    auto graphsMgm = zenoApp->graphsManagment();
    ZASSERT_EXIT(graphsMgm);
    IGraphsModel* pModel = graphsMgm->currentModel();
    ZASSERT_EXIT(pModel);
    const QModelIndex& index = pModel->indexFromPath(path);
    if (!index.isValid())
    {
        pModel->removeCommandParam(path);
        return;
    }
    const QString& paramName = index.data(ROLE_VPARAM_NAME).toString();
    QVariant val = index.data(ROLE_PARAM_VALUE);
    PARAM_CONTROL ctrl = (PARAM_CONTROL)index.data(ROLE_PARAM_CTRL).toInt();

    const QString& typeDesc = index.data(ROLE_PARAM_TYPE).toString();
    const QVariant& pros = index.data(ROLE_VPARAM_CTRL_PROPERTIES);

    int row = m_pTableWidget->rowCount();
    m_pTableWidget->insertRow(row);
    QTableWidgetItem* pParamItem = new QTableWidgetItem(name);
    pParamItem->setTextAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    m_pTableWidget->setItem(row, 0, pParamItem);
    pParamItem->setData(Qt::UserRole, path);

    QTableWidgetItem* pValueItem = new QTableWidgetItem();
    pValueItem->setFlags(pValueItem->flags() & ~Qt::ItemIsSelectable);
    m_pTableWidget->setItem(row, 1, pValueItem);
    pValueItem->setData(Qt::UserRole, val);

    CallbackCollection cbSet;
    cbSet.cbEditFinished = [=](QVariant newValue) {
        if (newValue != val)
            pValueItem->setData(Qt::UserRole, newValue);
    };

    QWidget* pControl = zenoui::createWidget(val, ctrl, typeDesc, cbSet, pros);
    m_pTableWidget->setCellWidget(row, 1, pControl);
}

void ZenoCommandParamsPanel::initTableWidget()
{
    auto graphsMgm = zenoApp->graphsManagment();
    ZASSERT_EXIT(graphsMgm);
    IGraphsModel* pModel = graphsMgm->currentModel();
    ZASSERT_EXIT(pModel);
    const FuckQMap<QString, QString>& params = pModel->commandParams();
    for (const auto& path : params.keys())
    {
        appendRow(path, params[path]);
    }
}

void ZenoCommandParamsPanel::onExport()
{
    if (m_pTableWidget->rowCount() <= 0)
        return;
    QString filePath = QFileDialog::getSaveFileName(this, "Path to Save", "", "Command Params File(*.json);; All Files(*);;");
    if (filePath.isEmpty())
        return;
    rapidjson::StringBuffer s;
    RAPIDJSON_WRITER writer(s);
    writer.StartObject();
    auto graphsMgm = zenoApp->graphsManagment();
    ZASSERT_EXIT(graphsMgm);
    IGraphsModel* pModel = graphsMgm->currentModel();
    ZASSERT_EXIT(pModel);
    for (int row = 0; row < m_pTableWidget->rowCount(); row++)
    {
        const auto* pItem = m_pTableWidget->item(row, 0);
        QString param = pItem->text();
        const QString& path = pItem->data(Qt::UserRole).toString();
        const QModelIndex& index = pModel->indexFromPath(path);
        if (index.isValid())
        {
            writer.Key(param.toUtf8());
            const QString& sockType = index.data(ROLE_PARAM_TYPE).toString();
            const auto* pValueItem = m_pTableWidget->item(row, 1);
            QVariant val = pValueItem->data(Qt::UserRole);
            JsonHelper::AddVariant(val, sockType, writer);
        }

    }
    writer.EndObject();

    QFile file(filePath);
    if (!file.open(QIODevice::WriteOnly)) {
        qWarning() << Q_FUNC_INFO << "Failed to open" << filePath << file.errorString();
        zeno::log_error("Failed to open file for write: {} ({})", filePath.toStdString(),
            file.errorString().toStdString());
        return;
    }

    file.write(s.GetString());
    file.close();
}

void ZenoCommandParamsPanel::onItemClicked(QTableWidgetItem* item)
{
    if (item->column() != 0)
        return;
    const QString& path = item->data(Qt::UserRole).toString();
    auto graphsMgm = zenoApp->graphsManagment();
    ZASSERT_EXIT(graphsMgm);
    IGraphsModel* pModel = graphsMgm->currentModel();
    ZASSERT_EXIT(pModel);
    const QString& ident = UiHelper::getSockNode(path);
    QModelIndex idx = pModel->nodeIndex(ident);
    if (idx.isValid())
    {
        QModelIndex subgIdx = idx.data(ROLE_SUBGRAPH_IDX).toModelIndex();
        const QString& subgName = subgIdx.data(ROLE_OBJNAME).toString();
        ZenoMainWindow* pWin = zenoApp->getMainWindow();
        ZASSERT_EXIT(pWin);
        ZenoGraphsEditor* pEditor = pWin->getAnyEditor();
        if (pEditor) {
            pEditor->activateTab(subgName, "", ident, false);
        }
    }
}

void ZenoCommandParamsPanel::onItemChanged(QTableWidgetItem* item)
{
    if (item->column() != 0)
        return;
    const QString& value = item->data(Qt::DisplayRole).toString();
    const QString& path = item->data(Qt::UserRole).toString();
    auto graphsMgm = zenoApp->graphsManagment();
    ZASSERT_EXIT(graphsMgm);
    IGraphsModel* pModel = graphsMgm->currentModel();
    ZASSERT_EXIT(pModel);
    const FuckQMap<QString, QString>& params = pModel->commandParams();
    if (!params.contains(path))
        return;
    const QString& oldVal = params[path];
    if (oldVal != value)
    {
        pModel->updateCommandParam(path, value);
    }
}

void ZenoCommandParamsPanel::onUpdateCommandParams(const QString& path)
{
    auto graphsMgm = zenoApp->graphsManagment();
    ZASSERT_EXIT(graphsMgm);
    IGraphsModel* pModel = graphsMgm->currentModel();
    ZASSERT_EXIT(pModel);
    const FuckQMap<QString, QString>& params = pModel->commandParams();
    for (int row = 0; row < m_pTableWidget->rowCount(); row++)
    {
        const QString& name = m_pTableWidget->item(row, 0)->data(Qt::DisplayRole).toString();
        const QString& itemPath = m_pTableWidget->item(row, 0)->data(Qt::UserRole).toString();
        if (path == itemPath)
        {
            if (params.contains(path)) // update
            {
                m_pTableWidget->item(row, 0)->setData(Qt::DisplayRole, params[path]);
            }
            else // remove
            {
                m_pTableWidget->removeRow(row);
            }
            return;
        }
    }
    //add
    appendRow(path, params[path]);
}

void ZenoCommandParamsPanel::onModelClear()
{
    while (m_pTableWidget->rowCount() > 0)
    {
        m_pTableWidget->removeRow(0);
    }
}

void ZenoCommandParamsPanel::onModelInited()
{
    onModelClear();
    auto graphsMgm = zenoApp->graphsManagment();
    if (!graphsMgm)
        return;
    IGraphsModel* pModel = graphsMgm->currentModel();
    if (!pModel)
        return;
    connect(pModel, &IGraphsModel::updateCommandParamSignal, this, &ZenoCommandParamsPanel::onUpdateCommandParams);
    connect(pModel, &IGraphsModel::modelClear, this, &ZenoCommandParamsPanel::onModelClear);
    initTableWidget();
}
