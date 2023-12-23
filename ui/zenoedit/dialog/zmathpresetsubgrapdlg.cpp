#include "zmathpresetsubgrapdlg.h"
#include <zeno/utils/logger.h>
#include <zenoui/style/zenostyle.h>
#include "zenoapplication.h"
#include <zenomodel/include/graphsmanagment.h>
#include <zenomodel/include/igraphsmodel.h>
#include <zenomodel/include/nodeparammodel.h>
#include <zenomodel/include/uihelper.h>
#include "variantptr.h"
#include "nodesview/zenographseditor.h"
#include "nodesys/zenosubgraphscene.h"
#include "zenomainwindow.h"
#include <zenoui/comctrl/zlabel.h>
#include <zenoui/comctrl/zcombobox.h>

//dialog
ZMatchPresetSubgraphDlg::ZMatchPresetSubgraphDlg(const QMap<QString, STMatchMatInfo>& info, QWidget* parent)
    : ZFramelessDialog(parent)
    , m_matchInfos(info)
{
    initUi(info);
    QString path = ":/icons/zeno-logo.png";
    this->setTitleIcon(QIcon(path));
    this->setTitleText(tr("Math Info"));
    resize(ZenoStyle::dpiScaledSize(QSize(500, 600)));
}

void ZMatchPresetSubgraphDlg::initUi(const QMap<QString, STMatchMatInfo>& info)
{
    QWidget* pWidget = new QWidget(this);
    QVBoxLayout* pLayout = new QVBoxLayout(pWidget);
    m_pTreeView = new QTreeView(this);
    m_pTreeView->setHeaderHidden(true);
    m_pModel = new QStandardItemModel(this);
    m_pTreeView->setModel(m_pModel);
    connect(m_pModel, &QStandardItemModel::rowsInserted, this, &ZMatchPresetSubgraphDlg::onRowInserted);
    initModel();
    pLayout->addWidget(m_pTreeView);
    QHBoxLayout* pHLayout = new QHBoxLayout(this);
    QPushButton* pOkBtn = new QPushButton(tr("Ok"), this);
    QPushButton* pCancelBtn = new QPushButton(tr("Cancel"), this);
    int width = ZenoStyle::dpiScaled(80);
    int height = ZenoStyle::dpiScaled(30);
    pOkBtn->setFixedSize(width, height);
    pCancelBtn->setFixedSize(width, height);
    pHLayout->addStretch();
    pHLayout->addWidget(pOkBtn);
    pHLayout->addWidget(pCancelBtn);
    pLayout->addLayout(pHLayout);
    this->setMainWidget(pWidget);

    connect(pOkBtn, &QPushButton::clicked, this, &ZMatchPresetSubgraphDlg::accept);
    connect(pCancelBtn, &QPushButton::clicked, this, &ZMatchPresetSubgraphDlg::reject);
}

void ZMatchPresetSubgraphDlg::initModel()
{
    for (const auto& key : m_matchInfos.keys())
    {
        const auto& info = m_matchInfos[key];
        QStandardItem* pItem = new QStandardItem(info.m_matType);
        pItem->setData(QVariant::fromValue(info.m_matKeys), Qt::UserRole);
        m_pModel->appendRow(pItem);
        for (const auto& key : info.m_matchInfo.keys())
        {
            const auto& match = info.m_matchInfo[key];
            QStandardItem* pChildItem = new QStandardItem;
            QWidget* pWidget = new QWidget;
            pChildItem->setData(key, Qt::DisplayRole);
            pItem->appendRow(pChildItem);
        }
        m_pTreeView->setExpanded(pItem->index(), true);
    }
}

void ZMatchPresetSubgraphDlg::onRowInserted(const QModelIndex& parent, int first, int last)
{
    const QModelIndex& idx = m_pModel->index(first, 0, parent);
    if (idx.isValid() && parent.isValid())
    {
        QWidget* pWidget = new QWidget(this);
        QHBoxLayout* pLayout = new QHBoxLayout(pWidget);
        pLayout->setMargin(0);
        QString param = idx.data(Qt::DisplayRole).toString();
        pLayout->addStretch();

        ZComboBox* pComboBox = new ZComboBox(this);
        QList<QString> lst = parent.data(Qt::UserRole).value<QSet<QString> >().toList();
        lst.prepend(param);
        pComboBox->addItems(lst);
        pComboBox->setCurrentText(param);
        pLayout->addWidget(pComboBox);
        m_pTreeView->setIndexWidget(idx, pWidget);
        connect(pComboBox, &ZComboBox::currentTextChanged, this, [=]() {
            QString mtlid = parent.data(Qt::DisplayRole).toString();
            if (m_matchInfos.contains(mtlid) && m_matchInfos[mtlid].m_matchInfo.contains(param))
            {
                m_matchInfos[mtlid].m_matchInfo[param] = pComboBox->currentText();
            }
        });
    }
}

QMap<QString, STMatchMatInfo> ZMatchPresetSubgraphDlg::getMatchInfo(const QMap<QString, STMatchMatInfo>& info, QWidget* parent)
{
    ZMatchPresetSubgraphDlg dlg(info, parent);
    if (dlg.exec() == QDialog::Accepted)
        return dlg.m_matchInfos;
    return QMap<QString, STMatchMatInfo>();
}