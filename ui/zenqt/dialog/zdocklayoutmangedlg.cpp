#include "ui_zdocklayoutmangedlg.h"
#include "zdocklayoutmangedlg.h"
#include <QSettings>
#include "settings/zsettings.h"
#include <QMessageBox>
#include <QInputDialog>

ZDockLayoutMangeDlg::ZDockLayoutMangeDlg(QWidget *parent)
    : QDialog(parent)
    , ui(new Ui::ZLayoutMangeDlgClass())
{
    ui->setupUi(this);
    initUI();
}

ZDockLayoutMangeDlg::~ZDockLayoutMangeDlg()
{
    delete ui;
}

void ZDockLayoutMangeDlg::initUI() {
    setWindowTitle(tr("Layout Manage"));
    QSettings settings(QSettings::UserScope, zsCompanyName, zsEditor);
    settings.beginGroup("layout");
    QStringList lst = settings.childGroups();
    if (lst.contains("LatestLayout"))
        lst.removeOne("LatestLayout");
    ui->m_listWidget->addItems(lst);

    ui->m_deleteButton->setEnabled(false);
    ui->m_renameButton->setEnabled(false);

    connect(ui->m_listWidget, &QListWidget::itemSelectionChanged, this, [=]() {
        bool isSelect = ui->m_listWidget->selectedItems().size() > 0;
        ui->m_deleteButton->setEnabled(isSelect);
        ui->m_renameButton->setEnabled(isSelect);
    });
    connect(ui->m_deleteButton, &QPushButton::clicked, this, [=]() {
        QString key = ui->m_listWidget->currentItem()->data(Qt::DisplayRole).toString();
        if (QMessageBox::question(this, tr("Delete"), tr("delete layout [%1]?").arg(key)) == QMessageBox::No)
            return;

        QSettings settings(QSettings::UserScope, zsCompanyName, zsEditor);
        settings.beginGroup("layout");
        settings.remove(key);
        settings.endGroup();
        ui->m_listWidget->takeItem(ui->m_listWidget->currentRow());
        emit layoutChangedSignal();
    });
    connect(ui->m_renameButton, &QPushButton::clicked, this, [=]() {
        QString key = ui->m_listWidget->currentItem()->data(Qt::DisplayRole).toString();
        QString newkey = QInputDialog::getText(this, tr("Rename"), tr("Name"), QLineEdit::Normal, key);
        if (newkey.isEmpty())
            return;

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
        ui->m_listWidget->currentItem()->setText(newkey);
        emit layoutChangedSignal();
    });
}