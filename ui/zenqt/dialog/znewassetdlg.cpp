#include "znewassetdlg.h"
#include "ui_znewassetdlg.h"
#include "zenoapplication.h"
#include "model/graphsmanager.h"
#include "model/assetsmodel.h"


ZNewAssetDlg::ZNewAssetDlg(QWidget* parent)
    : QDialog(parent)
{
    m_ui = new Ui::NewAssetDlg;
    m_ui->setupUi(this);

    connect(m_ui->editName, &QLineEdit::textChanged, [=](const QString& newValue) {
        QString dirPath = QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation);
        QString assetPath = dirPath + "/ZENO/assets/" + newValue + ".zda";
        m_ui->lblSavepath->setText(assetPath);
    });
}

zeno::AssetInfo ZNewAssetDlg::getAsset() const
{
    zeno::AssetInfo info;
    info.name = m_ui->editName->text().toStdString();
    info.path = m_ui->lblSavepath->text().toStdString();
    info.majorVer = m_ui->numMajorVer->value();
    info.minorVer = m_ui->numMinorVer->value();
    return info;
}

void ZNewAssetDlg::accept()
{
    const QString& editname = m_ui->editName->text();
    if (editname.isEmpty()) {
        QMessageBox::warning(this, tr("New Asset"), tr("the asset name cannot be empty").arg(editname));
        return;
    }

    AssetsModel* pModel = zenoApp->graphsManager()->assetsModel();
    if (pModel->getAssetGraph(editname)) {
        QMessageBox::warning(this, tr("New Asset"), tr("the asset with name {} has existed, please change another name").arg(editname));
    }
    else {
        QDialog::accept();
    }
}

void ZNewAssetDlg::reject()
{
    QDialog::reject();
}