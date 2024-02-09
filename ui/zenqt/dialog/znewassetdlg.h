#ifndef __ZNEW_ASSET_DLG_H__
#define __ZNEW_ASSET_DLG_H__

#include <QtWidgets>
#include <zeno/core/Assets.h>

namespace Ui
{
    class NewAssetDlg;
}

class ZNewAssetDlg : public QDialog
{
    Q_OBJECT
public:
    ZNewAssetDlg(QWidget* parent = nullptr);
    zeno::AssetInfo getAsset() const;

    void accept() override;
    void reject() override;

private:
    Ui::NewAssetDlg* m_ui;
};

#endif