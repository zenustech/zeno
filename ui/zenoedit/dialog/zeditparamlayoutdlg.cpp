#include "zeditparamlayoutdlg.h"
#include "ui_zeditparamlayoutdlg.h"
#include "zassert.h"
#include <zenomodel/include/uihelper.h>


ZEditParamLayoutDlg::ZEditParamLayoutDlg(ViewParamModel* pModel, QWidget* parent)
    : QDialog(parent)
    , m_proxyModel(nullptr)
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

    m_ctrls["Float"] = CONTROL_FLOAT;
    m_ctrls["String"] = CONTROL_STRING;
    m_ctrls["Boolean"] = CONTROL_BOOL;
    m_ctrls["Multiline String"] = CONTROL_MULTILINE_STRING;

    m_ui->listConctrl->addItems(lstCtrls);

    m_proxyModel = pModel;
    m_ui->paramsView->setModel(m_proxyModel);

    QItemSelectionModel* selModel = m_ui->paramsView->selectionModel();
    QModelIndex selIdx = selModel->currentIndex();
    const QModelIndex& wtfIdx = m_proxyModel->index(0, 0);
    selModel->setCurrentIndex(wtfIdx, QItemSelectionModel::SelectCurrent);
    m_ui->paramsView->expandAll();

    connect(m_ui->btnAdd, SIGNAL(clicked()), this, SLOT(onBtnAdd()));
}

void ZEditParamLayoutDlg::onBtnAdd()
{
    QModelIndex ctrlIdx = m_ui->listConctrl->currentIndex();
    if (!ctrlIdx.isValid())
        return;

    QModelIndex layerIdx = m_ui->paramsView->currentIndex();
    if (!layerIdx.isValid())
        return;

    QString ctrlName = ctrlIdx.data().toString();
    VPARAM_TYPE type = (VPARAM_TYPE)layerIdx.data(ROLE_VPARAM_TYPE).toInt();
    QStandardItem* pItem = m_proxyModel->itemFromIndex(layerIdx);
    ZASSERT_EXIT(pItem);
    QStringList existNames;
    for (int r = 0; r < pItem->rowCount(); r++)
    {
        QStandardItem* pChildItem = pItem->child(r);
        ZASSERT_EXIT(pChildItem);
        QString _name = pChildItem->data(ROLE_VPARAM_NAME).toString();
        existNames.append(_name);
    }

    if (ctrlName == "Tab")
    {
        if (type != VPARAM_ROOT)
        {
            QMessageBox::information(this, "Error", "create tab needs to place under the root");
            return;
        }
        QString newTabName = UiHelper::getUniqueName(existNames, "Tab");
        VParamItem* pNewItem = new VParamItem(VPARAM_TAB, newTabName);
        pItem->appendRow(pNewItem);
    }
    else if (ctrlName == "Group")
    {
        if (type != VPARAM_TAB && type != VPARAM_DEFAULT_TAB)
        {
            QMessageBox::information(this, "Error ", "create group needs to place under the tab");
            return;
        }
        QString newGroup = UiHelper::getUniqueName(existNames, "Group");
        VParamItem* pNewItem = new VParamItem(VPARAM_GROUP, newGroup);
        pItem->appendRow(pNewItem);
    }
    else
    {
        if (type != VPARAM_GROUP)
        {
            QMessageBox::information(this, "Error ", "create control needs to place under the group");
            return;
        }

        PARAM_CONTROL ctrl = CONTROL_NONE;
        if (m_ctrls.find(ctrlName) != m_ctrls.end())
            ctrl = m_ctrls[ctrlName];

        QString newItem = UiHelper::getUniqueName(existNames, "Param");
        VParamItem* pNewItem = new VParamItem(VPARAM_PARAM, newItem);
        pNewItem->ctrl = ctrl;
        pItem->appendRow(pNewItem);
    }
}