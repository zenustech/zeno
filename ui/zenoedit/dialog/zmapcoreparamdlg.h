#ifndef __ZMAP_COREPARAM_DLG_H__
#define __ZMAP_COREPARAM_DLG_H__

#include <QtWidgets>

namespace Ui
{
    class MapCoreparamDlg;
}

class ZMapCoreparamDlg : public QDialog
{
    Q_OBJECT
public:
    ZMapCoreparamDlg(const QPersistentModelIndex& idx, QWidget* parent = nullptr);
    QModelIndex coreIndex() const;

private:
    Ui::MapCoreparamDlg* m_ui;
    QStandardItemModel* m_model;
};

#endif