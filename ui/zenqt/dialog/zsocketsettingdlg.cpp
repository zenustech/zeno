#include "zsocketsettingdlg.h"
#include "style/zenostyle.h"
#include "widgets/zcombobox.h"
#include "util/uihelper.h"

ZSocketSettingDlg::ZSocketSettingDlg(const QModelIndexList& indexs, QWidget* parent)
    : ZFramelessDialog(parent)
    , m_indexs(indexs)
{
    QString path = ":/icons/zeno-logo.png";
    this->setTitleIcon(QIcon(path));
    this->setTitleText(tr("Sockets Settings"));
    m_mainWidget = new QWidget(this);
    QVBoxLayout* pLayout = new QVBoxLayout(m_mainWidget);
    pLayout->setAlignment(Qt::AlignTop);
    pLayout->setSpacing(ZenoStyle::dpiScaled(10));
    this->setMainWidget(m_mainWidget);
    resize(ZenoStyle::dpiScaledSize(QSize(300, 300)));
    initView();
    initButtons();
}

ZSocketSettingDlg::~ZSocketSettingDlg()
{
}


void ZSocketSettingDlg::initView()
{
    m_pModel = new QStandardItemModel(this);
    QTreeView* pTreeView = new QTreeView(m_mainWidget);
    pTreeView->setModel(m_pModel);
    pTreeView->setHeaderHidden(true);
    pTreeView->setMinimumHeight(ZenoStyle::dpiScaled(100));
    pTreeView->setMaximumHeight(ZenoStyle::dpiScaled(450));
    m_mainWidget->layout()->addWidget(pTreeView);

    for (const auto& index : m_indexs)
    {
        QString name = index.data(ROLE_PARAM_NAME).toString();
        QStandardItem* pItem = new QStandardItem(name);
        pItem->setData(index, Qt::UserRole);
        m_pModel->appendRow(pItem);

        int type = index.data(ROLE_SOCKET_TYPE).toInt();
        QStandardItem* pChildItem = new QStandardItem("Socket Type");
        pItem->appendRow(pChildItem);
        ZComboBox* pComboBox = new ZComboBox(this);
        pComboBox->setMinimumWidth(ZenoStyle::dpiScaled(100));
        pComboBox->addItem("ReadOnly", zeno::Socket_ReadOnly);
        pComboBox->addItem("Clone", zeno::Socket_Clone);
        pComboBox->addItem("Owning", zeno::Socket_Owning);
        for (int i = 0; i < pComboBox->count(); i++)
        {
            if (pComboBox->itemData(i).toInt() == type)
            {
                pComboBox->setCurrentIndex(i);
                pChildItem->setData(type, Qt::UserRole);
                break;
            }
        }
        connect(pComboBox, &ZComboBox::currentTextChanged, this, [=]() {
            int idx = pComboBox->currentIndex();
            auto val = pComboBox->itemData(idx);
            pChildItem->setData(val, Qt::UserRole);
        });
        QWidget* pWidget = new QWidget(this);
        QHBoxLayout* pLayout = new QHBoxLayout(pWidget);
        pLayout->setMargin(0);
        pLayout->addStretch();
        pLayout->addWidget(pComboBox);
        pTreeView->setIndexWidget(pChildItem->index(), pWidget);
        pTreeView->setExpanded(pItem->index(), true);
    }
}

void ZSocketSettingDlg::initButtons()
{
    int width = ZenoStyle::dpiScaled(80);
    int height = ZenoStyle::dpiScaled(30);
    QDialogButtonBox* pButtonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    m_mainWidget->layout()->addWidget(pButtonBox);
    connect(pButtonBox, &QDialogButtonBox::accepted, this, &ZSocketSettingDlg::onOKClicked);
    connect(pButtonBox, &QDialogButtonBox::rejected, this, &ZSocketSettingDlg::reject);
}

void ZSocketSettingDlg::onOKClicked()
{
    for (int row = 0; row < m_pModel->rowCount(); row++)
    {
        const QModelIndex& parent = m_pModel->index(row, 0);
        int row1 = 0;
        auto paramIdx = parent.data(Qt::UserRole).toModelIndex();
        while (parent.child(row1, 0).isValid())
        {
            int val = parent.child(row1, 0).data(Qt::UserRole).toInt();
            int oldVal = paramIdx.data(ROLE_SOCKET_TYPE).toInt();
            if (oldVal != val)
            {
                UiHelper::qIndexSetData(paramIdx, val, ROLE_SOCKET_TYPE);
            }
            row1++;
        }
    }
    accept();
}