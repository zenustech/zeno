#include "zeditparamlayoutdlg.h"
#include "ui_zeditparamlayoutdlg.h"


ZEditParamLayoutDlg::ZEditParamLayoutDlg(ViewParamModel* pModel, QWidget* parent)
    : QDialog(parent)
    , m_clone(nullptr)
{
    m_ui = new Ui::EditParamLayoutDlg;
    m_ui->setupUi(this);

    QStringList lstCtrls = {
        "Tab",
        "Group",
        "Integer",
        "Float",
        "String",
        "Boolean",
        "Multiline String",
        "Float Vector 4",
        "Float Vector 3",
        "Float Vector 2",
        "Integer Vector 4",
        "Integer Vector 3",
        "Integer Vector 2",
        "Color",
        "Curve",
    };

    m_ui->listConctrl->addItems(lstCtrls);
    m_ui->paramsView->setModel(pModel);
}