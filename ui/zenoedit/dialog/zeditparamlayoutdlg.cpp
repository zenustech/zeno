#include "zeditparamlayoutdlg.h"
#include "ui_zeditparamlayoutdlg.h"
#include "zassert.h"
#include <zenomodel/include/uihelper.h>


static CONTROL_ITEM_INFO controlList[] = {
    {"Integer",             CONTROL_INT,    "int"},
    {"Float",               CONTROL_FLOAT,  "float"},
    {"String",              CONTROL_STRING, "string"},
    {"Boolean",             CONTROL_BOOL,   "bool"},
    {"Multiline String",    CONTROL_MULTILINE_STRING, "string"},
    {"Float Vector 4",      CONTROL_VEC4_FLOAT, "vec4f"},
    {"Float Vector 3",      CONTROL_VEC3_FLOAT, "vec3f"},
    {"Float Vector 2",      CONTROL_VEC2_FLOAT, "vec2f"},
    {"Integer Vector 4",    CONTROL_VEC4_INT,   "vec4i"},
    {"Integer Vector 3",    CONTROL_VEC3_INT,   "vec3i"},
    {"Integer Vector 2",    CONTROL_VEC2_INT,   "vec2i"},
    {"Color",   CONTROL_COLOR,  "color"},
    {"Curve",   CONTROL_CURVE,  "curve"},
};

static QString getControlName(PARAM_CONTROL ctrl)
{
    for (int i = 0; i < sizeof(controlList) / sizeof(CONTROL_ITEM_INFO); i++)
    {
        if (controlList[i].ctrl == ctrl)
        {
            return controlList[i].name;
        }
    }
    return "";
}

static PARAM_CONTROL getControlByName(const QString& name)
{
    for (int i = 0; i < sizeof(controlList) / sizeof(CONTROL_ITEM_INFO); i++)
    {
        if (controlList[i].name == name)
        {
            return controlList[i].ctrl;
        }
    }
    return CONTROL_NONE;
}


ParamTreeItemDelegate::ParamTreeItemDelegate(QObject* parent)
    : QStyledItemDelegate(parent)
{
}

ParamTreeItemDelegate::~ParamTreeItemDelegate()
{
}

QWidget* ParamTreeItemDelegate::createEditor(QWidget* parent, const QStyleOptionViewItem& option, const QModelIndex& index) const
{
    bool bEditable = index.data(ROLE_VAPRAM_EDITTABLE).toBool();
    if (!bEditable)
        return nullptr;
    return QStyledItemDelegate::createEditor(parent, option, index);
}



ZEditParamLayoutDlg::ZEditParamLayoutDlg(ViewParamModel* pModel, QWidget* parent)
    : QDialog(parent)
    , m_model(pModel)
    , m_proxyModel(nullptr)
{
    m_ui = new Ui::EditParamLayoutDlg;
    m_ui->setupUi(this);

    QStringList lstCtrls = {
        "Tab",
        "Group"
    };
    for (int i = 0; i < sizeof(controlList) / sizeof(CONTROL_ITEM_INFO); i++)
    {
        lstCtrls.append(controlList[i].name);
        m_ui->cbControl->addItem(controlList[i].name);
    }

    m_ui->listConctrl->addItems(lstCtrls);

    m_proxyModel = new ViewParamModel(this);
    m_proxyModel->clone(m_model);

    m_ui->paramsView->setModel(m_proxyModel);
    m_ui->paramsView->setItemDelegate(new ParamTreeItemDelegate(m_ui->paramsView));

    QItemSelectionModel* selModel = m_ui->paramsView->selectionModel();
    QModelIndex selIdx = selModel->currentIndex();
    const QModelIndex& wtfIdx = m_proxyModel->index(0, 0);
    selModel->setCurrentIndex(wtfIdx, QItemSelectionModel::SelectCurrent);
    m_ui->paramsView->expandAll();

    connect(selModel, SIGNAL(currentChanged(const QModelIndex&, const QModelIndex&)),
            this, SLOT(onTreeCurrentChanged(const QModelIndex&, const QModelIndex&)));
    connect(m_ui->editName, SIGNAL(editingFinished()), this, SLOT(onNameEditFinished()));
    connect(m_ui->btnAdd, SIGNAL(clicked()), this, SLOT(onBtnAdd()));
    connect(m_ui->btnApply, SIGNAL(clicked()), this, SLOT(onApply()));
    connect(m_ui->btnOk, SIGNAL(clicked()), this, SLOT(onOk()));
    connect(m_ui->btnCancel, SIGNAL(clicked()), this, SLOT(onCancel()));
}

void ZEditParamLayoutDlg::onTreeCurrentChanged(const QModelIndex& current, const QModelIndex& previous)
{
    QStandardItem* pCurrentItem = m_proxyModel->itemFromIndex(current);
    if (!pCurrentItem)  return;

    const QString& name = pCurrentItem->data(ROLE_VPARAM_NAME).toString();
    m_ui->editName->setText(name);
    bool bEditable = pCurrentItem->data(ROLE_VAPRAM_EDITTABLE).toBool();
    m_ui->editName->setEnabled(bEditable);

    VPARAM_TYPE type = (VPARAM_TYPE)pCurrentItem->data(ROLE_VPARAM_TYPE).toInt();
    if (type == VPARAM_TAB)
    {
        m_ui->cbControl->setEnabled(false);
    }
    else if (type == VPARAM_GROUP)
    {
        m_ui->cbControl->setEnabled(false);
    }
    else if (type == VPARAM_PARAM)
    {
        if (!bEditable)
        {
            m_ui->cbControl->setEnabled(false);
            return;
        }

        const QString& paramName = name;
        PARAM_CONTROL ctrl = (PARAM_CONTROL)pCurrentItem->data(ROLE_PARAM_CTRL).toInt();
        const QString& ctrlName = getControlName(ctrl);

        m_ui->cbControl->setEnabled(true);
        m_ui->cbControl->setCurrentText(ctrlName);
    }
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
        if (type != VPARAM_TAB)
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
        QString typeDesc;
        for (int i = 0; i < sizeof(controlList) / sizeof(CONTROL_ITEM_INFO); i++)
        {
            if (controlList[i].name == ctrlName)
            {
                ctrl = controlList[i].ctrl;
                typeDesc = controlList[i].defaultType;
                break;
            }
        }

        QString newItem = UiHelper::getUniqueName(existNames, "Param");
        VParamItem* pNewItem = new VParamItem(VPARAM_PARAM, newItem);
        pNewItem->m_info.control = ctrl;
        pNewItem->m_info.typeDesc = typeDesc;
        pItem->appendRow(pNewItem);
    }
}

void ZEditParamLayoutDlg::onNameEditFinished()
{
    const QModelIndex& currIdx = m_ui->paramsView->currentIndex();
    if (!currIdx.isValid())
        return;

    QStandardItem* pItem = m_proxyModel->itemFromIndex(currIdx);
    pItem->setData(m_ui->editName->text(), ROLE_VPARAM_NAME);
}

void ZEditParamLayoutDlg::onLabelEditFinished()
{

}

void ZEditParamLayoutDlg::onHintEditFinished()
{

}

void ZEditParamLayoutDlg::onApply()
{
    m_model->clone(m_proxyModel);
}

void ZEditParamLayoutDlg::onOk()
{
    accept();
    onApply();
}

void ZEditParamLayoutDlg::onCancel()
{
    reject();
}
