#ifndef __ZEDIT_PARAM_LAYOUT_DLG_H__
#define __ZEDIT_PARAM_LAYOUT_DLG_H__

#include <QtWidgets>
#include <zenomodel/include/viewparammodel.h>

namespace Ui
{
    class EditParamLayoutDlg;
}

class ZEditParamLayoutDlg : public QDialog
{
    Q_OBJECT
public:
    ZEditParamLayoutDlg(ViewParamModel* pModel, QWidget* parent = nullptr);

private slots:
    void onBtnAdd();

private:
    ViewParamModel* m_proxyModel;

    QMap<QString, PARAM_CONTROL> m_ctrls;

    Ui::EditParamLayoutDlg* m_ui;
};



#endif