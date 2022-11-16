#include "zeditparamlayoutdlg.h"
#include "ui_zeditparamlayoutdlg.h"
#include "zassert.h"
#include <zenomodel/include/uihelper.h>
#include "zmapcoreparamdlg.h"


static CONTROL_ITEM_INFO controlList[] = {
    {"Integer",             CONTROL_INT,            "int"},
    {"Float",               CONTROL_FLOAT,          "float"},
    {"String",              CONTROL_STRING,         "string"},
    {"Boolean",             CONTROL_BOOL,           "bool"},
    {"Multiline String",    CONTROL_MULTILINE_STRING, "string"},
    {"read path",           CONTROL_READPATH,       "string"},
    {"write path",          CONTROL_WRITEPATH,      "string"},
    {"Enum",                CONTROL_ENUM,           "string"},
    {"Float Vector 4",      CONTROL_VEC4_FLOAT,     "vec4f"},
    {"Float Vector 3",      CONTROL_VEC3_FLOAT,     "vec3f"},
    {"Float Vector 2",      CONTROL_VEC2_FLOAT,     "vec2f"},
    {"Integer Vector 4",    CONTROL_VEC4_INT,       "vec4i"},
    {"Integer Vector 3",    CONTROL_VEC3_INT,       "vec3i"},
    {"Integer Vector 2",    CONTROL_VEC2_INT,       "vec2i"},
    {"Color",               CONTROL_COLOR,          "color"},
    {"Curve",               CONTROL_CURVE,          "curve"},
    {"SpinBox",             CONTROL_HSPINBOX,       "int"},
    {"Slider",              CONTROL_HSLIDER,        "int"},
    {"SpinBoxSlider",       CONTROL_SPINBOX_SLIDER, "int"},
};

static CONTROL_ITEM_INFO getControl(PARAM_CONTROL ctrl)
{
    for (int i = 0; i < sizeof(controlList) / sizeof(CONTROL_ITEM_INFO); i++)
    {
        if (controlList[i].ctrl == ctrl)
        {
            return controlList[i];
        }
    }
    return CONTROL_ITEM_INFO();
}

static CONTROL_ITEM_INFO getControlByName(const QString& name)
{
    for (int i = 0; i < sizeof(controlList) / sizeof(CONTROL_ITEM_INFO); i++)
    {
        if (controlList[i].name == name)
        {
            return controlList[i];
        }
    }
    return CONTROL_ITEM_INFO();
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
    //if (!bEditable)
    //    return nullptr;
    return QStyledItemDelegate::createEditor(parent, option, index);
}



ZEditParamLayoutDlg::ZEditParamLayoutDlg(ViewParamModel* pModel, bool bNodeUI, const QPersistentModelIndex& nodeIdx, QWidget* parent)
    : QDialog(parent)
    , m_model(pModel)
    , m_proxyModel(nullptr)
    , m_index(nodeIdx)
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

    m_proxyModel = new ViewParamModel(bNodeUI, m_model->nodeIdx(), this);
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

    QShortcut* shortcut = new QShortcut(QKeySequence(Qt::Key_Delete), m_ui->paramsView);
    connect(shortcut, SIGNAL(activated()), this, SLOT(onParamTreeDeleted()));

    connect(m_ui->btnChooseCoreParam, SIGNAL(clicked(bool)), this, SLOT(onChooseParamClicked()));
    connect(m_ui->editComboItems, SIGNAL(editingFinished()), this, SLOT(onComboItemsEditFinished()));
    connect(m_ui->editMin, SIGNAL(editingFinished()), this, SLOT(onMinEditFinished()));
    connect(m_ui->editMax, SIGNAL(editingFinished()), this, SLOT(onMaxEditFinished()));
    connect(m_ui->editStep, SIGNAL(editingFinished()), this, SLOT(onStepEditFinished()));
}

void ZEditParamLayoutDlg::onParamTreeDeleted()
{
    QModelIndex idx = m_ui->paramsView->currentIndex();
    if (!idx.isValid()) return;
    m_proxyModel->removeRow(idx.row(), idx.parent());
}

void ZEditParamLayoutDlg::onTreeCurrentChanged(const QModelIndex& current, const QModelIndex& previous)
{
    QStandardItem* pCurrentItem = m_proxyModel->itemFromIndex(current);
    if (!pCurrentItem)  return;

    const QString& name = pCurrentItem->data(ROLE_VPARAM_NAME).toString();
    m_ui->editName->setText(name);
    bool bEditable = pCurrentItem->data(ROLE_VAPRAM_EDITTABLE).toBool();
    //m_ui->editName->setEnabled(bEditable);

    VPARAM_TYPE type = (VPARAM_TYPE)pCurrentItem->data(ROLE_VPARAM_TYPE).toInt();
    if (type == VPARAM_TAB)
    {
        m_ui->cbControl->setEnabled(false);
        m_ui->stackProperties->setCurrentIndex(0);
    }
    else if (type == VPARAM_GROUP)
    {
        m_ui->cbControl->setEnabled(false);
        m_ui->stackProperties->setCurrentIndex(0);
    }
    else if (type == VPARAM_PARAM)
    {
        const QString& paramName = name;
        PARAM_CONTROL ctrl = (PARAM_CONTROL)pCurrentItem->data(ROLE_PARAM_CTRL).toInt();
        const QString& ctrlName = getControl(ctrl).name;

        m_ui->cbControl->setEnabled(true);
        m_ui->cbControl->setCurrentText(ctrlName);

        const QString& coreName = pCurrentItem->data(ROLE_PARAM_NAME).toString();
        const QString& coreType = pCurrentItem->data(ROLE_PARAM_TYPE).toString();
        PARAM_CLASS coreCls = (PARAM_CLASS)pCurrentItem->data(ROLE_PARAM_SOCKETTYPE).toInt();

        m_ui->editCoreParamName->setText(coreName);
        m_ui->editCoreParamType->setText(coreType);

        if (ctrl == CONTROL_ENUM)
        {
            CONTROL_PROPERTIES pros = pCurrentItem->data(ROLE_VPARAM_CTRL_PROPERTIES).value<CONTROL_PROPERTIES>();
            if (pros.find("items") != pros.end())
            {
                m_ui->editComboItems->setText(pros["items"].toStringList().join(","));
            }
            m_ui->stackProperties->setCurrentIndex(1);
        }
        else if (ctrl == CONTROL_HSLIDER ||
                 ctrl == CONTROL_HSPINBOX ||
                 ctrl == CONTROL_SPINBOX_SLIDER)
        {
            m_ui->stackProperties->setCurrentIndex(2);
        }
        else
        {
            m_ui->stackProperties->setCurrentIndex(0);
        }
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
        CONTROL_ITEM_INFO ctrl = getControlByName(ctrlName);
        QString newItem = UiHelper::getUniqueName(existNames, ctrl.name);
        VParamItem* pNewItem = new VParamItem(VPARAM_PARAM, newItem);
        pNewItem->m_info.control = ctrl.ctrl;
        pNewItem->m_info.typeDesc = ctrl.defaultType;
        pNewItem->m_info.value = UiHelper::initVariantByControl(ctrl.ctrl);
        pItem->appendRow(pNewItem);
    }
}

void ZEditParamLayoutDlg::onNameEditFinished()
{
    const QModelIndex& currIdx = m_ui->paramsView->currentIndex();
    if (!currIdx.isValid())
        return;

    QStandardItem* pItem = m_proxyModel->itemFromIndex(currIdx);
    pItem->setData(m_ui->editName->text(), ROLE_VPARAM_NAME);   //call: VParamItem::setData
}

void ZEditParamLayoutDlg::onLabelEditFinished()
{

}

void ZEditParamLayoutDlg::onHintEditFinished()
{

}

void ZEditParamLayoutDlg::onComboItemsEditFinished()
{
    QModelIndex layerIdx = m_ui->paramsView->currentIndex();
    if (!layerIdx.isValid() && layerIdx.data(ROLE_VPARAM_TYPE) != VPARAM_PARAM)
        return;

    const QString& items = m_ui->editComboItems->text();
    QStringList lst = items.split(",");
    if (lst.isEmpty())
        return;

    CONTROL_PROPERTIES properties = layerIdx.data(ROLE_VPARAM_CTRL_PROPERTIES).value<CONTROL_PROPERTIES>();
    properties["items"] = lst;
    m_proxyModel->setData(layerIdx, properties, ROLE_VPARAM_CTRL_PROPERTIES);
}

void ZEditParamLayoutDlg::onMinEditFinished()
{
    QModelIndex layerIdx = m_ui->paramsView->currentIndex();
    if (!layerIdx.isValid() && layerIdx.data(ROLE_VPARAM_TYPE) != VPARAM_PARAM)
        return;

    CONTROL_PROPERTIES properties = layerIdx.data(ROLE_VPARAM_CTRL_PROPERTIES).value<CONTROL_PROPERTIES>();
    int from = m_ui->editMin->text().toInt();
    properties["min"] = from;
    m_proxyModel->setData(layerIdx, properties, ROLE_VPARAM_CTRL_PROPERTIES);
}

void ZEditParamLayoutDlg::onMaxEditFinished()
{
    QModelIndex layerIdx = m_ui->paramsView->currentIndex();
    if (!layerIdx.isValid() && layerIdx.data(ROLE_VPARAM_TYPE) != VPARAM_PARAM)
        return;

    CONTROL_PROPERTIES properties = layerIdx.data(ROLE_VPARAM_CTRL_PROPERTIES).value<CONTROL_PROPERTIES>();
    int to = m_ui->editMax->text().toInt();
    properties["max"] = to;
    m_proxyModel->setData(layerIdx, properties, ROLE_VPARAM_CTRL_PROPERTIES);
}

void ZEditParamLayoutDlg::onStepEditFinished()
{
    QModelIndex layerIdx = m_ui->paramsView->currentIndex();
    if (!layerIdx.isValid() && layerIdx.data(ROLE_VPARAM_TYPE) != VPARAM_PARAM)
        return;

    CONTROL_PROPERTIES properties = layerIdx.data(ROLE_VPARAM_CTRL_PROPERTIES).value<CONTROL_PROPERTIES>();
    int step = m_ui->editStep->text().toInt();
    properties["step"] = step;
    m_proxyModel->setData(layerIdx, properties, ROLE_VPARAM_CTRL_PROPERTIES);
}

void ZEditParamLayoutDlg::onChooseParamClicked()
{
    ZMapCoreparamDlg dlg(m_index);
    if (QDialog::Accepted == dlg.exec())
    {
        QModelIndex coreIdx = dlg.coreIndex();

        QModelIndex viewparamIdx = m_ui->paramsView->currentIndex();
        QStandardItem* pItem = m_proxyModel->itemFromIndex(viewparamIdx);
        VParamItem* pViewItem = static_cast<VParamItem*>(pItem);

        if (coreIdx.isValid())
        {
            pViewItem->mapCoreParam(coreIdx);

            //update control info.
            const QString& newName = coreIdx.data(ROLE_PARAM_NAME).toString();
            const QString& typeDesc = coreIdx.data(ROLE_PARAM_TYPE).toString();

            m_ui->editCoreParamName->setText(newName);
            m_ui->editCoreParamType->setText(typeDesc);

            PARAM_CONTROL newCtrl = UiHelper::getControlType(typeDesc);
            pViewItem->setData(newCtrl, ROLE_PARAM_CTRL);
        }
        else
        {
            pViewItem->m_index = QModelIndex();
            pViewItem->setData(CONTROL_NONE, ROLE_PARAM_CTRL);
            m_ui->editCoreParamName->setText("");
            m_ui->editCoreParamType->setText("");
        }
    }
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
