
#include "zshortcutsettingdlg.h"
#include <QSettings>
#include "settings/zsettings.h"
#include <QVBoxLayout>
#include <rapidjson/document.h>
#include <QPushButton>
#include <zenoui/style/zenostyle.h>
#include <zenomodel/include/jsonhelper.h>
#include "settings/zenosettingsmanager.h"

ZShortCutSettingDlg::ZShortCutSettingDlg(QWidget *parent) :
    QDialog(parent), 
    m_pTableWidget(nullptr) 
{
    initUI();
}

ZShortCutSettingDlg::~ZShortCutSettingDlg()
{
}

bool ZShortCutSettingDlg::eventFilter(QObject *obj, QEvent *event) 
{
    if (event->type() == QEvent::KeyPress) 
    {
        if (obj == m_pTableWidget) {
            if (QWidget *widget = m_pTableWidget->cellWidget(m_pTableWidget->currentRow(), m_pTableWidget->currentColumn())) {
                if (QLineEdit *lineEdit = qobject_cast<QLineEdit *>(widget)) {
                    lineEdit->installEventFilter(this);
                    QKeyEvent *keyEvent = static_cast<QKeyEvent *>(event);
                    if (keyEvent->key() >= Qt::Key_F1 && keyEvent->key() <= Qt::Key_F12) {
                        lineEdit->setText(QKeySequence(keyEvent->key()).toString());
                        return true;
                    }
                }
            }
        } else if (QLineEdit *lineEdit = qobject_cast<QLineEdit *>(obj)) {
            QKeyEvent *keyEvent = static_cast<QKeyEvent *>(event);
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
            lineEdit->setText(qsKey);
            return true;
        }
    }
    return QDialog::eventFilter(obj, event);
}

void ZShortCutSettingDlg::initUI() {
    QVBoxLayout *pLayout = new QVBoxLayout(this);
    m_pTableWidget = new QTableWidget(this);
    m_pTableWidget->setColumnCount(2);
    m_pTableWidget->setColumnWidth(0, ZenoStyle::dpiScaled(130));
    m_pTableWidget->installEventFilter(this);
    m_pTableWidget->verticalHeader()->setVisible(false);
    QStringList labels = QStringList() << tr("Description") << tr("ShortCut");
    m_pTableWidget->setHorizontalHeaderLabels(labels);
    m_pTableWidget->horizontalHeader()->setDefaultAlignment(Qt::AlignLeft | Qt::AlignVCenter);
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

    connect(pOKButton, &QPushButton::clicked, this, [=]() {
        writeShortCutInfo();
        accept();
    });

    connect(pCancelButton, &QPushButton::clicked, this, [=]() {
        reject();
    });

    //init table widget
    m_shortCutInfos = ZenoSettingsManager::GetInstance().getValue(zsShortCut).value<QVector<ShortCutInfo>>();
    m_pTableWidget->setRowCount(m_shortCutInfos.size());
    int row = 0;
    for (auto info : m_shortCutInfos) 
    {
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

    connect(m_pTableWidget, &QTableWidget::doubleClicked, this, [=]() {
        if (QWidget *widget = m_pTableWidget->cellWidget(m_pTableWidget->currentRow(), m_pTableWidget->currentColumn())) {
            if (QLineEdit *lineEdit = qobject_cast<QLineEdit *>(widget)) {
                lineEdit->installEventFilter(this);
            }
        }
    });
    this->resize(ZenoStyle::dpiScaled(280), ZenoStyle::dpiScaled(500));
    this->setWindowTitle(tr("Shortcut Setting"));

}

void ZShortCutSettingDlg::writeShortCutInfo() {
    rapidjson::StringBuffer str;
    PRETTY_WRITER writer(str);
    writer.StartArray();
    bool bChanged = false;
    for (auto info : m_shortCutInfos) {
        writer.StartObject();
        writer.Key("key");
        writer.String(info.key.toUtf8());
        writer.Key("shortcut");
        writer.String(info.shortcut.toUtf8());
        writer.EndObject();
        if (ZenoSettingsManager::GetInstance().getShortCut(info.key) != info.shortcut) 
        {
            ZenoSettingsManager::GetInstance().setShortCut(info.key, info.shortcut);
            bChanged = true;
        }
    }
    writer.EndArray();
    if (bChanged) {
        QString strJson = QString::fromUtf8(str.GetString());
        ZenoSettingsManager::GetInstance().setValue(zsShortCut, strJson);
    }
}
