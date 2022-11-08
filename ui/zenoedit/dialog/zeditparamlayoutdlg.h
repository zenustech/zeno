#ifndef __ZEDIT_PARAM_LAYOUT_DLG_H__
#define __ZEDIT_PARAM_LAYOUT_DLG_H__

#include <QtWidgets>
#include <zenomodel/include/viewparammodel.h>

namespace Ui
{
    class EditParamLayoutDlg;
}

struct CONTROL_ITEM_INFO
{
    QString name;
    PARAM_CONTROL ctrl;
    QString defaultType;
};


class ZEditParamLayoutDlg : public QDialog
{
    Q_OBJECT
public:
    ZEditParamLayoutDlg(ViewParamModel* pModel, QWidget* parent = nullptr);

private slots:
    void onBtnAdd();
    void onApply();
    void onOk();
    void onCancel();

private:
    ViewParamModel* m_proxyModel;
    ViewParamModel* m_model;

    Ui::EditParamLayoutDlg* m_ui;
};



#endif