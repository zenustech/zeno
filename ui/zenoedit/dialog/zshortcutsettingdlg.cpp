
#include "zshortcutsettingdlg.h"
#include <QSettings>
#include "settings/zsettings.h"
#include <QVBoxLayout>
#include <rapidjson/document.h>
#include <QPushButton>
#include <zenoui/style/zenostyle.h>
#include <zenomodel/include/jsonhelper.h>
#include "settings/zenosettingsmanager.h"
#include <zenoui/comctrl/zcombobox.h>
#include <zenoui/comctrl/zlineedit.h>

ZShortCutItemDelegate::ZShortCutItemDelegate(QObject* parent) : _base(parent)
{

}
// editing
QWidget* ZShortCutItemDelegate::createEditor(QWidget* parent,
    const QStyleOptionViewItem& option,
    const QModelIndex& index) const
{
    QString text = index.data().toString();
    ZLineEdit* pLineEdit = new ZLineEdit(text, parent);
    pLineEdit->setProperty("key", index.sibling(index.row(), 0).data(Qt::DisplayPropertyRole));
    pLineEdit->installEventFilter(parent);
    return pLineEdit;
}

bool ZShortCutItemDelegate::eventFilter(QObject* object, QEvent* event)
{
    if (ZLineEdit* lineEdit = qobject_cast<ZLineEdit*>(object))
    {
        if (event->type() == QEvent::ContextMenu)
        {
            QString key = lineEdit->property("key").toString();
            if (key != ShortCut_MovingView && key != ShortCut_RotatingView && key != ShortCut_ScalingView)
                return false;
            QMenu* pMenu = new QMenu;
            QStringList list = { "Mouse L" , "Mouse M", "Mouse R", "Mouse S" };
            for (const auto& text : list)
            {
                QAction* pAtion = new QAction(text, pMenu);
                pMenu->addAction(pAtion);
            }
            connect(pMenu, &QMenu::triggered, this, [=](QAction* action) {
                QString text = lineEdit->text();
            if (!text.isEmpty() && !text.endsWith("+"))
            {
                text += "+";
            }
            text += action->text();
            lineEdit->setText(text);
            });
            pMenu->exec(QCursor::pos());
            pMenu->deleteLater();
            return true;
        }
        else if (event->type() == QEvent::KeyPress)
        {
            QKeyEvent* keyEvent = static_cast<QKeyEvent*>(event);
            int uKey = keyEvent->key();
            Qt::Key key = static_cast<Qt::Key>(uKey);
            if (key == Qt::Key_Control || key == Qt::Key_Shift || key == Qt::Key_Alt || key == Qt::Key_Enter ||
                key == Qt::Key_Return || key == Qt::Key_CapsLock || key == Qt::Key_Escape ||
                key == Qt::Key_Backspace || key == Qt::Key_unknown) {
                return false;
            }
            Qt::KeyboardModifiers modifiers = keyEvent->modifiers();
            if (modifiers & Qt::ShiftModifier) {
                uKey += Qt::SHIFT;
            }
            if (modifiers & Qt::ControlModifier) {
                uKey += Qt::CTRL;
            }
            if (modifiers & Qt::AltModifier) {
                uKey += Qt::ALT;
            }
            QString qsKey = QKeySequence(uKey).toString();
            if (modifiers == Qt::NoModifier)
            {
                QString selectedText = lineEdit->selectedText();
                qsKey = lineEdit->text().replace(selectedText, "") + qsKey;
            }
            //if (keyEvent->key() >= Qt::Key_F1 && keyEvent->key() <= Qt::Key_F12) {
            //    qsKey = QKeySequence(keyEvent->key()).toString();
            //}
            lineEdit->setText(qsKey);
            return true;
        }
    }
    return _base::eventFilter(object, event);
}

//dialog
ZShortCutSettingDlg::ZShortCutSettingDlg(QWidget *parent) :
    QDialog(parent), 
    m_pTableWidget(nullptr)
{
    initUI();
}

ZShortCutSettingDlg::~ZShortCutSettingDlg()
{
}


void ZShortCutSettingDlg::initUI()
{
    QVBoxLayout *pLayout = new QVBoxLayout(this);
    QHBoxLayout* pHLayout_top = new QHBoxLayout;
    ZComboBox* pComboBox = new ZComboBox(this);
    pComboBox->setFixedWidth(ZenoStyle::dpiScaled(100));
    pComboBox->addItems(QStringList() << tr("Default") << tr("Houdini") << tr("Maya"));
    pComboBox->setCurrentIndex(ZenoSettingsManager::GetInstance().getValue(zsShortCutStyle).toInt());
    pHLayout_top->addWidget(pComboBox);
    pHLayout_top->addSpacerItem(new QSpacerItem(10, 10, QSizePolicy::Expanding));
    pLayout->addLayout(pHLayout_top);

    m_pTableWidget = new QTableWidget(this);
    m_pTableWidget->horizontalHeader()->setStretchLastSection(true);
    m_pTableWidget->setColumnCount(2);
    m_pTableWidget->setColumnWidth(0, ZenoStyle::dpiScaled(130));
    ZShortCutItemDelegate* pDelegate = new ZShortCutItemDelegate(m_pTableWidget);
    m_pTableWidget->setItemDelegate(pDelegate);
    m_pTableWidget->verticalHeader()->setVisible(false);
    QStringList labels = QStringList() << tr("Description") << tr("ShortCut");
    m_pTableWidget->setHorizontalHeaderLabels(labels);
    m_pTableWidget->horizontalHeader()->setDefaultAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    m_pTableWidget->setMouseTracking(true);
    pLayout->addWidget(m_pTableWidget);
    QHBoxLayout *pHLayout = new QHBoxLayout;
    QPushButton *pOKButton = new QPushButton(tr("OK"), this);
    QPushButton *pCancelButton = new QPushButton(tr("Cancel"), this);
    qreal width = ZenoStyle::dpiScaled(80);
    pOKButton->setFixedWidth(width);
    pCancelButton->setFixedWidth(width);
    pHLayout->addWidget(pOKButton);
    pHLayout->addWidget(pCancelButton);
    pHLayout->setAlignment(Qt::AlignRight);
    pLayout->addLayout(pHLayout);

    connect(pComboBox, &ZComboBox::_textActivated, this, [=]() {
        onCurrentIndexChanged(pComboBox->currentIndex());
    });

    connect(pOKButton, &QPushButton::clicked, this, [=]() {
        ZenoSettingsManager::GetInstance().writeShortCutInfo(m_shortCutInfos, pComboBox->currentIndex());
        accept();
    });

    connect(pCancelButton, &QPushButton::clicked, this, [=]() {
        reject();
    });

    //init table widget
    m_shortCutInfos = ZenoSettingsManager::GetInstance().getValue(zsShortCut).value<QVector<ShortCutInfo>>();
    m_pTableWidget->setRowCount(m_shortCutInfos.size());
    int row = 0;
    for (auto info : m_shortCutInfos) {
        QTableWidgetItem *descItem = new QTableWidgetItem(info.desc);
        descItem->setFlags(descItem->flags() & (~Qt::ItemFlag::ItemIsEditable));
        descItem->setData(Qt::DisplayPropertyRole, info.key);
        m_pTableWidget->setItem(row, 0, descItem);
        m_pTableWidget->setItem(row, 1, new QTableWidgetItem(info.shortcut));
        row++;
    }
    connect(m_pTableWidget, &QTableWidget::itemChanged, this, [=](QTableWidgetItem *item) {
        if (item->column() != 1)
            return;
        int row = item->row();
        QString key = m_pTableWidget->item(row, 0)->data(Qt::DisplayPropertyRole).toString();
        for (auto &shortcutInfo : m_shortCutInfos) {
            if (shortcutInfo.key == key) {
                shortcutInfo.shortcut = item->data(Qt::DisplayRole).toString();
                break;
            }
        }
    });

    this->resize(ZenoStyle::dpiScaled(280), ZenoStyle::dpiScaled(500));
    this->setWindowTitle(tr("Shortcut Setting"));
}

void ZShortCutSettingDlg::onCurrentIndexChanged(int index)
{
    while (m_pTableWidget->rowCount() > 0)
    {
        m_pTableWidget->removeRow(0);
    }
    m_shortCutInfos = ZenoSettingsManager::GetInstance().getDefaultShortCutInfo(index);
    m_pTableWidget->setRowCount(m_shortCutInfos.size());
    int row = 0;
    for (auto info : m_shortCutInfos) {
        QTableWidgetItem* descItem = new QTableWidgetItem(info.desc);
        descItem->setFlags(descItem->flags() & (~Qt::ItemFlag::ItemIsEditable));
        descItem->setData(Qt::DisplayPropertyRole, info.key);
        m_pTableWidget->setItem(row, 0, descItem);
        m_pTableWidget->setItem(row, 1, new QTableWidgetItem(info.shortcut));
        row++;
    }
}