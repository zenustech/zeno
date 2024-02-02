#include "zpreferencesdlg.h"
#include <zenoui/style/zenostyle.h>
#include <zenoui/comctrl/zpathedit.h>
#include "settings/zenosettingsmanager.h"
#include "zenoapplication.h"
#include "zenomainwindow.h"
#include "startup/zstartup.h"
#include "zshortcutsettingdlg.h"
#include <zenoui/comctrl/zlabel.h>

//Language Pane
ZLanguagePane::ZLanguagePane(QWidget* parent)
    :QWidget(parent)
{
    bool bChinese = ZenoSettingsManager::GetInstance().getValue(zsUseChinese).toBool();
    QHBoxLayout* pLayout = new QHBoxLayout(this);
    m_pEngBtn = new QRadioButton(tr("English"), this);
    m_pCNBtn = new QRadioButton(tr("Chinese"), this);
    m_pCNBtn->setChecked(bChinese);
    m_pEngBtn->setChecked(!bChinese);
    pLayout->addWidget(m_pEngBtn);
    pLayout->addWidget(m_pCNBtn);
    pLayout->setAlignment(Qt::AlignLeft | Qt::AlignTop);
}

void ZLanguagePane::saveValue()
{
    bool value = m_pCNBtn->isChecked();
    auto& inst = ZenoSettingsManager::GetInstance();
    bool bChinese = inst.getValue(zsUseChinese).toBool();
    if (inst.setValue(zsUseChinese, value))
    {
        QMessageBox msg(QMessageBox::Information, tr("Language"),
            tr("Please restart Zeno to apply changes."),
            QMessageBox::Ok, this);
        msg.exec();
    }
}

//NASLOC Pane
ZNASLOCPane::ZNASLOCPane(QWidget* parent)
    :QWidget(parent)
{
    const QString& path = ZenoSettingsManager::GetInstance().getValue(zsNASLOC).toString();
    QHBoxLayout* pLayout = new QHBoxLayout(this);
    CALLBACK_SWITCH cbSwitch = [=](bool bOn) {
        zenoApp->getMainWindow()->setInDlgEventLoop(bOn); //deal with ubuntu dialog slow problem when update viewport.
    };
    m_pPthEdit = new ZPathEdit(cbSwitch, path, this);
    pLayout->addWidget(new QLabel("Set NASLOC"));
    pLayout->addWidget(m_pPthEdit);
    pLayout->addItem(new QSpacerItem(10, 10, QSizePolicy::Expanding));
    pLayout->setAlignment(Qt::AlignLeft | Qt::AlignTop);
}

void ZNASLOCPane::saveValue()
{
    auto& inst = ZenoSettingsManager::GetInstance();
    const QString& oldPath = inst.getValue(zsNASLOC).toString();
    QString path = m_pPthEdit->text();
    path.replace('\\', '/');
    if (inst.setValue(zsNASLOC, path))
    {
        startUp(true);
    }
}

//ZenoCache Pane
ZenoCachePane::ZenoCachePane(QWidget* parent) : QWidget(parent)
{
    auto& inst = ZenoSettingsManager::GetInstance();

    QVariant varEnableCache = inst.getValue(zsCacheEnable);
    QVariant varTempCacheDir = inst.getValue(zsCacheAutoRemove);
    QVariant varCacheRoot = inst.getValue(zsCacheDir);
    QVariant varCacheNum = inst.getValue(zsCacheNum);
    QVariant varAutoCleanCache = inst.getValue(zsCacheAutoClean);
    QVariant varEnableShiftChangeFOV = inst.getValue(zsEnableShiftChangeFOV);
    QVariant varViewportPointSizeScale = inst.getValue(zsViewportPointSizeScale);

    bool bEnableCache = varEnableCache.isValid() ? varEnableCache.toBool() : false;
    bool bTempCacheDir = varTempCacheDir.isValid() ? varTempCacheDir.toBool() : false;
    QString cacheRootDir = varCacheRoot.isValid() ? varCacheRoot.toString() : "";
    int cacheNum = varCacheNum.isValid() ? varCacheNum.toInt() : 1;
    double viewportPointSizeScale = varViewportPointSizeScale.isValid() ? varViewportPointSizeScale.toDouble() : 1;
    bool bAutoCleanCache = varAutoCleanCache.isValid() ? varAutoCleanCache.toBool() : true;
    bool bEnableShiftChangeFOV = varEnableShiftChangeFOV.isValid() ? varEnableShiftChangeFOV.toBool() : true;

    CALLBACK_SWITCH cbSwitch = [=](bool bOn) {
        zenoApp->getMainWindow()->setInDlgEventLoop(bOn); //deal with ubuntu dialog slow problem when update viewport.
    };
    m_pPathEdit = new ZPathEdit(cbSwitch, cacheRootDir);
    m_pPathEdit->setFixedWidth(256);
    m_pPathEdit->setEnabled(!bTempCacheDir && bEnableCache);
    m_pTempCacheDir = new QCheckBox;
    m_pTempCacheDir->setCheckState(bTempCacheDir ? Qt::Checked : Qt::Unchecked);
    m_pTempCacheDir->setEnabled(bEnableCache);
    m_pAutoCleanCache = new QCheckBox;
    m_pAutoCleanCache->setCheckState(bAutoCleanCache ? Qt::Checked : Qt::Unchecked);
    m_pAutoCleanCache->setEnabled(bEnableCache && !bTempCacheDir);

    m_pEnableShiftChangeFOV = new QCheckBox;
    m_pEnableShiftChangeFOV->setCheckState(bEnableShiftChangeFOV ? Qt::Checked : Qt::Unchecked);

    connect(m_pTempCacheDir, &QCheckBox::stateChanged, [=](bool state) {
        m_pPathEdit->setText("");
        m_pPathEdit->setEnabled(!state);
        m_pAutoCleanCache->setChecked(Qt::Unchecked);
        m_pAutoCleanCache->setEnabled(!state);
    });

    m_pCacheNumSpinBox = new QSpinBox;
    m_pCacheNumSpinBox->setRange(1, 10000);
    m_pCacheNumSpinBox->setValue(cacheNum);
    m_pCacheNumSpinBox->setEnabled(bEnableCache);

    m_pViewportPointSizeScaleSpinBox = new QDoubleSpinBox;
    m_pViewportPointSizeScaleSpinBox->setValue(viewportPointSizeScale);

    m_pEnableCheckbox = new QCheckBox;
    m_pEnableCheckbox->setCheckState(bEnableCache ? Qt::Checked : Qt::Unchecked);
    connect(m_pEnableCheckbox, &QCheckBox::stateChanged, [=](bool state) {
        if (!state)
        {
            m_pCacheNumSpinBox->clear();
            m_pPathEdit->clear();
            m_pTempCacheDir->setCheckState(Qt::Unchecked);
            m_pAutoCleanCache->setCheckState(Qt::Unchecked);
        }
        m_pCacheNumSpinBox->setEnabled(state);
        m_pPathEdit->setEnabled(state);
        m_pTempCacheDir->setEnabled(state);
        m_pAutoCleanCache->setEnabled(state && !m_pTempCacheDir->isChecked());
    });

    QGridLayout* pLayout = new QGridLayout(this);
    pLayout->addWidget(new QLabel(tr("Enable cache")), 0, 0);
    pLayout->addWidget(m_pEnableCheckbox, 0, 1);
    pLayout->addWidget(new QLabel(tr("Cache num")), 1, 0);
    pLayout->addWidget(m_pCacheNumSpinBox, 1, 1);
    pLayout->addWidget(new QLabel(tr("Temp cache directory")), 2, 0);
    pLayout->addWidget(m_pTempCacheDir, 2, 1);
    pLayout->addWidget(new QLabel(tr("Cache root")), 3, 0);
    pLayout->addWidget(m_pPathEdit, 3, 1);
    pLayout->addWidget(new QLabel(tr("Cache auto clean up")), 4, 0);
    pLayout->addWidget(m_pAutoCleanCache, 4, 1);
    pLayout->addWidget(new QLabel(tr("Enable Shift change FOV")), 5, 0);
    pLayout->addWidget(m_pEnableShiftChangeFOV, 5, 1);
    pLayout->addWidget(new QLabel(tr("Viewport Point Size scale")), 6, 0);
    pLayout->addWidget(m_pViewportPointSizeScaleSpinBox, 6, 1);
    QSpacerItem* pSpacerItem = new QSpacerItem(10, 10, QSizePolicy::Expanding);
    pLayout->addItem(pSpacerItem, 0, 2, 5);
    pLayout->setAlignment(Qt::AlignLeft | Qt::AlignTop);
}

void ZenoCachePane::saveValue()
{
    auto& inst = ZenoSettingsManager::GetInstance();
    inst.setValue(zsCacheEnable, m_pEnableCheckbox->checkState() == Qt::Checked);
    inst.setValue(zsCacheAutoRemove, m_pTempCacheDir->checkState() == Qt::Checked);
    inst.setValue(zsCacheDir, m_pPathEdit->text());
    inst.setValue(zsCacheNum, m_pCacheNumSpinBox->value());
    inst.setValue(zsCacheAutoClean, m_pAutoCleanCache->checkState() == Qt::Checked);
    inst.setValue(zsEnableShiftChangeFOV, m_pEnableShiftChangeFOV->checkState() == Qt::Checked);
    inst.setValue(zsViewportPointSizeScale, m_pViewportPointSizeScaleSpinBox->value());
}

//layout pane
ZLayoutPane::ZLayoutPane(QWidget* parent) : QWidget(parent)
{
    QHBoxLayout* pLayout = new QHBoxLayout(this);
    m_listWidget = new QListWidget(this);
    m_listWidget->setSelectionMode(QAbstractItemView::ExtendedSelection);
    pLayout->addWidget(m_listWidget);
    pLayout->setAlignment(Qt::AlignLeft | Qt::AlignTop);

    QSettings settings(QSettings::UserScope, zsCompanyName, zsEditor);
    settings.beginGroup("layout");
    QStringList lst = settings.childGroups();
    if (lst.contains("LatestLayout"))
        lst.removeOne("LatestLayout");
    if (lst.isEmpty())
        return;
    m_listWidget->addItems(lst);
    for (int i = 0; i < m_listWidget->count(); i++)
    {
        if (QListWidgetItem* pItem = m_listWidget->item(i))
            pItem->setData(Qt::UserRole, pItem->text());
    }

    QPushButton* deleteButton = new QPushButton(tr("Delete"), this);
    deleteButton->setEnabled(false);
    QPushButton* renameButton = new QPushButton(tr("Rename"), this);
    renameButton->setEnabled(false);

    QVBoxLayout *pVLayout = new QVBoxLayout;
    pVLayout->addWidget(renameButton);
    pVLayout->addWidget(deleteButton);
    pVLayout->setAlignment(Qt::AlignLeft | Qt::AlignTop);
    pLayout->addLayout(pVLayout);

    connect(m_listWidget, &QListWidget::itemSelectionChanged, this, [=]() {
        int count = m_listWidget->selectedItems().size();
        deleteButton->setEnabled(count > 0);
        renameButton->setEnabled(count == 1);
    });
    connect(deleteButton, &QPushButton::clicked, this, [=]() {
        QList<QListWidgetItem*> lst = m_listWidget->selectedItems();
        for (const auto& pItem : lst)
        {
            QString key = pItem->data(Qt::UserRole).toString();
            int row = m_listWidget->row(pItem);
            m_listWidget->takeItem(row);
            const char* prop = "delItems";
            QStringList lst = m_listWidget->property(prop).toStringList();
            lst << key;
            m_listWidget->setProperty(prop, lst);
        }
    });

    connect(renameButton, &QPushButton::clicked, this, [=]() {
        QListWidgetItem* pItem = m_listWidget->currentItem();
        if (!pItem)
            return;
        const QString& key = pItem->data(Qt::DisplayRole).toString();
        QString newkey = QInputDialog::getText(this, tr("Rename"), tr("Name"), QLineEdit::Normal, key);
        if (newkey.isEmpty())
            return;
        pItem->setData(Qt::DisplayRole, newkey);
    });
}

void ZLayoutPane::saveValue()
{
    bool bChanged = false;
    //delete layout
    const char* prop = "delItems";
    QStringList lst = m_listWidget->property(prop).toStringList();
    if (!lst.isEmpty())
    {
        bChanged = true;
        for (const auto& delKey : lst)
        {
            QSettings settings(QSettings::UserScope, zsCompanyName, zsEditor);
            settings.beginGroup("layout");
            settings.remove(delKey);
            settings.endGroup();
        }
    }

    //rename layout
    for (int i = 0; i < m_listWidget->count(); i++)
    {
        QListWidgetItem* pItem = m_listWidget->item(i);
        const QString& newkey = pItem->data(Qt::DisplayRole).toString();
        const QString& key = pItem->data(Qt::UserRole).toString();
        if (key != newkey)
        {
            QSettings settings(QSettings::UserScope, zsCompanyName, zsEditor);
            settings.beginGroup("layout");
            settings.beginGroup(key);
            QVariant value = settings.value("content");
            settings.endGroup();
            settings.remove(key);
            settings.beginGroup(newkey);
            settings.setValue("content", value);
            settings.endGroup();
            settings.endGroup();
            bChanged = true;
        }
    }
    if (bChanged)
    {
        zenoApp->getMainWindow()->loadSavedLayout();
    }
}

//TabBar
QSize ZPreferencesTabBar::tabSizeHint(int index) const {
    QSize s = QTabBar::tabSizeHint(index);
    s.transpose();
    return s;
}

void ZPreferencesTabBar::paintEvent(QPaintEvent* event) {
    QStylePainter painter(this);
    QStyleOptionTab opt;

    for (int i = 0; i < count(); i++)
    {
        initStyleOption(&opt, i);
        painter.drawControl(QStyle::CE_TabBarTabShape, opt);
        painter.save();

        QSize s = opt.rect.size();
        s.transpose();
        QRect r(QPoint(), s);
        r.moveCenter(opt.rect.center());
        opt.rect = r;

        QPoint c = tabRect(i).center();
        painter.translate(c);
        painter.rotate(90);
        painter.translate(-c);
        painter.drawControl(QStyle::CE_TabBarTabLabel, opt);
        painter.restore();
    }
}

ZPreferencesTabWidget::ZPreferencesTabWidget(QWidget* parent)
{

    initUI();
}

void ZPreferencesTabWidget::initUI()
{
    setTabBar(new ZPreferencesTabBar());
    setTabPosition(QTabWidget::West);
    setProperty("cssClass", "preferences");
    //language
    m_pLanguagePane = new ZLanguagePane(this);
    addTab(m_pLanguagePane, tr("Language"));
    //NASLOC pane
    m_pNASLOCPane = new ZNASLOCPane(this);
    addTab(m_pNASLOCPane, tr("NASLOC"));
    //ZenoCache pane
    m_pZenoCachePane = new ZenoCachePane(this);
    addTab(m_pZenoCachePane, tr("Zeno Cache"));
    //shortcut panel
    m_pShortcutsPane = new ShortcutsPane(this);
    addTab(m_pShortcutsPane, tr("Shortcuts"));
    //layout manage
    m_pLayoutPane = new ZLayoutPane(this);
    addTab(m_pLayoutPane, tr("Layout Manage"));
}

void ZPreferencesTabWidget::saveSettings()
{
    //language
    m_pLanguagePane->saveValue();
    //NASLOC
    m_pNASLOCPane->saveValue();
    //zeno cache
    m_pZenoCachePane->saveValue();
    //shortcuts
    m_pShortcutsPane->saveValue();
    //layout
    m_pLayoutPane->saveValue();
}

ZPreferencesDlg::ZPreferencesDlg(QWidget* parent) : ZFramelessDialog(parent), m_pTabWidget(nullptr)
{
    initUI();
}

ZPreferencesDlg::~ZPreferencesDlg()
{
}

void ZPreferencesDlg::initUI()
{
    QWidget* pMainWidget = new QWidget(this);
    m_pTabWidget = new ZPreferencesTabWidget(this);
    m_pTabWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    m_pOKBtn = new QPushButton(tr("OK"), this);
    m_pCancelBtn = new QPushButton(tr("Cancel"), this);
    QHBoxLayout* pHLyout = new QHBoxLayout();
    pHLyout->addWidget(m_pOKBtn);
    pHLyout->addWidget(m_pCancelBtn);
    pHLyout->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    QVBoxLayout* pLyout = new QVBoxLayout(pMainWidget);
    int margin = ZenoStyle::dpiScaled(10);
    pLyout->setContentsMargins(ZenoStyle::dpiScaled(2), margin, margin, margin);
    pLyout->addWidget(m_pTabWidget);
    pLyout->addLayout(pHLyout);
    pMainWidget->setLayout(pLyout);
    setTitleText(tr("Preferences"));
    setMainWidget(pMainWidget);
    resize(ZenoStyle::dpiScaledSize(QSize(800, 800)));
    setTitleIcon(QIcon(":/icons/toolbar_localSetting_idle.svg"));
    connect(m_pOKBtn, &QPushButton::clicked, this, [=]() {
        m_pTabWidget->saveSettings();
        accept();
    });
    connect(m_pCancelBtn, &QPushButton::clicked, this, &ZPreferencesDlg::reject);
}
