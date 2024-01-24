#include "zeditparamlayoutdlg.h"
#include "ui_zeditparamlayoutdlg.h"
#include "zassert.h"
#include "util/uihelper.h"
#include "zmapcoreparamdlg.h"
#include "util/uihelper.h"
#include "zenoapplication.h"
#include "model/graphsmanager.h"
#include "model/parammodel.h"
#include "widgets/zwidgetfactory.h"
#include "util/globalcontrolmgr.h"
#include "variantptr.h"
#include "iotags.h"
#include "style/zenostyle.h"
#include "widgets/zspinboxslider.h"
#include <zeno/utils/helper.h>


static CONTROL_ITEM_INFO controlList[] = {
    {"Tab",                 zeno::NullControl,  zeno::Param_Null,   ":/icons/parameter_control_tab.svg"},
    {"Group",               zeno::NullControl,  zeno::Param_Null,   ":/icons/parameter_control_group.svg"},
    {"Integer",             zeno::Lineedit,     zeno::Param_Int,    ":/icons/parameter_control_integer.svg"},
    {"Float",               zeno::Lineedit,     zeno::Param_Float,  ":/icons/parameter_control_float.svg"},
    {"String",              zeno::Lineedit,     zeno::Param_String, ":/icons/parameter_control_string.svg"},
    {"Boolean",             zeno::Checkbox,     zeno::Param_Bool,   ":/icons/parameter_control_boolean.svg"},
    {"Multiline String",    zeno::Multiline,    zeno::Param_String, ":/icons/parameter_control_string.svg"},
    {"read path",           zeno::Pathedit,     zeno::Param_String, ":/icons/parameter_control_fold.svg"},
    {"write path",          zeno::Pathedit,     zeno::Param_String, ":/icons/parameter_control_fold.svg"},
    {"Enum",                zeno::Combobox,     zeno::Param_String, ":/icons/parameter_control_enum.svg"},
    {"Float Vector 4",      zeno::Vec4edit,     zeno::Param_Vec4f,  ":/icons/parameter_control_floatVector4.svg"},
    {"Float Vector 3",      zeno::Vec3edit,     zeno::Param_Vec3f,  ":/icons/parameter_control_floatVector3.svg"},
    {"Float Vector 2",      zeno::Vec2edit,     zeno::Param_Vec2f,  ":/icons/parameter_control_floatVector2.svg"},
    {"Integer Vector 4",    zeno::Vec4edit,     zeno::Param_Vec4i,  ":/icons/parameter_control_integerVector4.svg"},
    {"Integer Vector 3",    zeno::Vec3edit,     zeno::Param_Vec3i,  ":/icons/parameter_control_integerVector3.svg"},
    {"Integer Vector 2",    zeno::Vec2edit,     zeno::Param_Vec2i,  ":/icons/parameter_control_integerVector2.svg"},
    {"Color",               zeno::Heatmap,      zeno::Param_Heatmap,":/icons/parameter_control_color.svg"},
    {"Pure Color",          zeno::Color,        zeno::Param_Vec3f,  ":/icons/parameter_control_color.svg"},
    {"Color Vec3f",         zeno::ColorVec,     zeno::Param_Vec3f,  ":/icons/parameter_control_color.svg"},
    {"Curve",               zeno::CurveEditor,  zeno::Param_Curve,  ":/icons/parameter_control_curve.svg"},
    {"SpinBox",             zeno::SpinBox,      zeno::Param_Int,    ":/icons/parameter_control_spinbox.svg"},
    {"DoubleSpinBox",       zeno::DoubleSpinBox,zeno::Param_Float,  ":/icons/parameter_control_spinbox.svg"},
    {"Slider",              zeno::Slider,       zeno::Param_Int,    ":/icons/parameter_control_slider.svg"},
    {"SpinBoxSlider",       zeno::SpinBoxSlider,zeno::Param_Int,    ":/icons/parameter_control_slider.svg"},
    {"Divider",             zeno::Seperator,    zeno::Param_Null,   ":/icons/parameter_control_divider.svg"},
};

static CONTROL_ITEM_INFO getControl(zeno::ParamControl ctrl, zeno::ParamType type)
{
    for (int i = 0; i < sizeof(controlList) / sizeof(CONTROL_ITEM_INFO); i++)
    {
        if (controlList[i].ctrl == ctrl && controlList[i].type == type)
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

static zeno::ParamType getTypeByControlName(const QString& name)
{
    for (int i = 0; i < sizeof(controlList) / sizeof(CONTROL_ITEM_INFO); i++)
    {
        if (controlList[i].name == name)
        {
            return controlList[i].type;
        }
    }
    return zeno::Param_Null;
}



ParamTreeItemDelegate::ParamTreeItemDelegate(ParamsModel* model, QObject *parent)
    : QStyledItemDelegate(parent)
    , m_model(model) 
{
}

ParamTreeItemDelegate::~ParamTreeItemDelegate()
{
}

QWidget* ParamTreeItemDelegate::createEditor(QWidget* parent, const QStyleOptionViewItem& option, const QModelIndex& index) const
{
    bool bEditable = false;// m_model->isEditable(index);
    if (!bEditable)
        return nullptr;
    return QStyledItemDelegate::createEditor(parent, option, index);
}

void ParamTreeItemDelegate::setModelData(QWidget *editor, QAbstractItemModel *model, const QModelIndex &index) const 
{
    QStyledItemDelegate::setModelData(editor, model, index);
    /*
    QString oldName = index.data().toString();
    QString newName = editor->property(editor->metaObject()->userProperty().name()).toString();
    if (oldName != newName) {
        int dstRow = 0;
        VParamItem *pTargetGroup = static_cast<VParamItem *>(m_model->itemFromIndex(index.parent()));
        if (pTargetGroup && !pTargetGroup->getItem(newName, &dstRow)) {
            QStyledItemDelegate::setModelData(editor, model, index);
        } else {
            QMessageBox::information(nullptr, tr("Info"), tr("The param name already exists"));
        }
    }
    */
}

void ParamTreeItemDelegate::paint(QPainter *painter, const QStyleOptionViewItem &option,
                                  const QModelIndex &index) const {
    QStyledItemDelegate::paint(painter, option, index);
}


ZEditParamLayoutDlg::ZEditParamLayoutDlg(ParamsModel* pModel, QWidget* parent)
    : QDialog(parent)
    , m_model(pModel)
    , m_bSubgraphNode(false)
    , m_paramsLayoutM(nullptr)
{
    m_ui = new Ui::EditParamLayoutDlg;
    m_ui->setupUi(this);
    initUI();

    m_ui->cbControl->addItem("");
    for (int i = 0; i < sizeof(controlList) / sizeof(CONTROL_ITEM_INFO); i++)
    {
        QListWidgetItem *item = new QListWidgetItem(controlList[i].name, m_ui->listConctrl);
        item->setIcon(QIcon(controlList[i].icon));
        m_ui->listConctrl->addItem(item);
        if (controlList[i].ctrl == zeno::NullControl)
            continue;
        m_ui->cbControl->addItem(controlList[i].name);
    }

    m_model = qobject_cast<ParamsModel*>(pModel);
    ZASSERT_EXIT(m_model);

    _initLayoutModel();
    initIcon(m_paramsLayoutM->invisibleRootItem());

    m_ui->paramsView->setModel(m_paramsLayoutM);
    //m_ui->paramsView->setItemDelegate(new ParamTreeItemDelegate(m_model, m_ui->paramsView));

    QItemSelectionModel* selModel = m_ui->paramsView->selectionModel();
    connect(selModel, SIGNAL(currentChanged(const QModelIndex &, const QModelIndex &)), this,
            SLOT(onTreeCurrentChanged(const QModelIndex &, const QModelIndex &)));
    QModelIndex selIdx = selModel->currentIndex();
    const QModelIndex& wtfIdx = m_paramsLayoutM->index(0, 0);
    selModel->setCurrentIndex(wtfIdx, QItemSelectionModel::SelectCurrent);
    m_ui->paramsView->expandAll();

    connect(m_ui->editName, SIGNAL(editingFinished()), this, SLOT(onNameEditFinished()));
    connect(m_ui->editLabel, SIGNAL(editingFinished()), this, SLOT(onLabelEditFinished()));
    connect(m_ui->btnAdd, SIGNAL(clicked()), this, SLOT(onBtnAdd()));
    connect(m_ui->btnApply, SIGNAL(clicked()), this, SLOT(onApply()));
    connect(m_ui->btnOk, SIGNAL(clicked()), this, SLOT(onOk()));
    connect(m_ui->btnCancel, SIGNAL(clicked()), this, SLOT(onCancel()));

    connect(m_paramsLayoutM, SIGNAL(editNameChanged(const QModelIndex&, const QString&, const QString&)),
            this, SLOT(onProxyItemNameChanged(const QModelIndex&, const QString&, const QString&)));

    QShortcut* shortcut = new QShortcut(QKeySequence(Qt::Key_Delete), m_ui->paramsView);
    connect(shortcut, SIGNAL(activated()), this, SLOT(onParamTreeDeleted()));

    connect(m_ui->btnChooseCoreParam, SIGNAL(clicked(bool)), this, SLOT(onChooseParamClicked()));
    connect(m_ui->editMin, SIGNAL(editingFinished()), this, SLOT(onMinEditFinished()));
    connect(m_ui->editMax, SIGNAL(editingFinished()), this, SLOT(onMaxEditFinished()));
    connect(m_ui->editStep, SIGNAL(editingFinished()), this, SLOT(onStepEditFinished()));
    connect(m_ui->cbControl, SIGNAL(currentIndexChanged(int)), this, SLOT(onControlItemChanged(int)));

    m_ui->itemsTable->setHorizontalHeaderLabels({ tr("Item Name") });
    connect(m_ui->itemsTable, SIGNAL(cellChanged(int, int)), this, SLOT(onComboTableItemsCellChanged(int, int)));

    m_ui->m_pUpButton->setFixedWidth(32);
    m_ui->m_pUpButton->setEnabled(false);
    m_ui->m_pUpButton->setIcon(QIcon(":/icons/moveUp.svg"));
    connect(m_ui->itemsTable, &QTableWidget::itemSelectionChanged, this, [=]() {
        m_ui->m_pUpButton->setEnabled(true);
        auto item = m_ui->itemsTable->currentItem();
        if (item) {
            int row = item->row();
            if (row == 0) {
                m_ui->m_pUpButton->setEnabled(false);
            }
        } else {
            m_ui->m_pUpButton->setEnabled(false);
        }
    });

    connect(m_ui->m_pUpButton, &QPushButton::clicked, this, [=]() {
        auto item = m_ui->itemsTable->currentItem();
        if (item) {
            int row = item->row() - 1;
            disconnect(m_ui->itemsTable, SIGNAL(cellChanged(int, int)), this,
                       SLOT(onComboTableItemsCellChanged(int, int)));
            QString text = item->text();
            item->setText(m_ui->itemsTable->item(row, 0)->text());
            connect(m_ui->itemsTable, SIGNAL(cellChanged(int, int)), this,
                    SLOT(onComboTableItemsCellChanged(int, int)));
            m_ui->itemsTable->item(row, 0)->setText(text);
            m_ui->itemsTable->setCurrentItem(m_ui->itemsTable->item(row, 0));
        }
    });

    connect(m_paramsLayoutM, &QStandardItemModel::dataChanged, this, &ZEditParamLayoutDlg::onViewParamDataChanged);
}

void ZEditParamLayoutDlg::initUI() 
{
    m_ui->labelCreateControl->setProperty("cssClass", "bold");
    m_ui->labelPrameter->setProperty("cssClass", "bold");
    m_ui->labelSetting->setProperty("cssClass", "bold");
    m_ui->label_11->setProperty("cssClass", "bold");
    m_ui->btnChooseCoreParam->setProperty("cssClass", "CoreParam");
    m_ui->listConctrl->setFixedWidth(ZenoStyle::dpiScaled(296));
    m_ui->paramsView->setFixedWidth(ZenoStyle::dpiScaled(296));
    m_ui->labelSetting->setMinimumWidth(ZenoStyle::dpiScaled(296));
    m_ui->btnAdd->setFixedSize(ZenoStyle::dpiScaled(66), ZenoStyle::dpiScaled(36));
    QSize buttonSize(ZenoStyle::dpiScaled(100), ZenoStyle::dpiScaled(36));
    m_ui->btnApply->setFixedSize(buttonSize);
    m_ui->btnCancel->setFixedSize(buttonSize);
    m_ui->btnOk->setFixedSize(buttonSize);
    m_ui->line->setLineWidth(2);
    m_ui->horizontalLayout_3->setSpacing(ZenoStyle::dpiScaled(20));
    m_ui->horizontalLayout_3->setContentsMargins(0, ZenoStyle::dpiScaled(8), 0, ZenoStyle::dpiScaled(8));
    m_ui->verticalLayout_2->setContentsMargins(ZenoStyle::dpiScaled(10), 0, 0, 0);
    m_ui->gridLayout->setVerticalSpacing(ZenoStyle::dpiScaled(8));
    m_ui->listConctrl->setAlternatingRowColors(true);
    m_ui->paramsView->setAlternatingRowColors(true);
    m_ui->listConctrl->setFocusPolicy(Qt::NoFocus);
    m_ui->paramsView->setFocusPolicy(Qt::NoFocus);
    resize(ZenoStyle::dpiScaled(900), ZenoStyle::dpiScaled(620));
    setFocusPolicy(Qt::ClickFocus);
}

void ZEditParamLayoutDlg::_initLayoutModel()
{
    //temp: temp layout model for params, will be replaced by specific proxy model.
    if (m_paramsLayoutM) {
        m_paramsLayoutM->clear();
        delete m_paramsLayoutM;
        m_paramsLayoutM = nullptr;
    }

    m_paramsLayoutM = new QStandardItemModel(this);

    QStandardItem* pRoot = new QStandardItem("root");
    QStandardItem* pInputs = new QStandardItem("input");
    QStandardItem* pOutputs = new QStandardItem("output");
    for (int r = 0; r < m_model->rowCount(); r++) {
        QModelIndex paramIdx = m_model->index(r);
        bool bInput = paramIdx.data(ROLE_ISINPUT).toBool();
        zeno::ParamInfo info = paramIdx.data(ROLE_PARAM_INFO).value<zeno::ParamInfo>();
        QStandardItem* paramItem = new QStandardItem(QString::fromStdString(info.name));
        const QString& paramName = QString::fromStdString(info.name);
        paramItem->setData(paramName, Qt::DisplayRole);
        paramItem->setData(paramName, ROLE_PARAM_NAME);
        paramItem->setData(paramName, ROLE_MAP_TO_PARAMNAME);
        paramItem->setData(UiHelper::zvarToQVar(info.defl), ROLE_PARAM_VALUE);
        paramItem->setData(info.control, ROLE_PARAM_CONTROL);
        paramItem->setData(info.type, ROLE_PARAM_TYPE);
        paramItem->setData(bInput, ROLE_ISINPUT);
        if (bInput)
            pInputs->appendRow(paramItem);
        else
            pOutputs->appendRow(paramItem);
        paramItem->setData(VPARAM_PARAM, ROLE_ELEMENT_TYPE);
    }
    pRoot->setData(VPARAM_TAB, ROLE_ELEMENT_TYPE);
    pInputs->setData(VPARAM_GROUP, ROLE_ELEMENT_TYPE);
    pOutputs->setData(VPARAM_GROUP, ROLE_ELEMENT_TYPE);

    pRoot->appendRow(pInputs);
    pRoot->appendRow(pOutputs);

    m_paramsLayoutM->appendRow(pRoot);
}

void ZEditParamLayoutDlg::initIcon(QStandardItem *pItem) 
{
    ZASSERT_EXIT(pItem);

    for (int r = 0; r < pItem->rowCount(); r++) 
    {
        QStandardItem *newItem = pItem->child(r);
        newItem->setData(getIcon(newItem), Qt::DecorationRole);
        if (newItem->rowCount() > 0)
            initIcon(newItem);
    }
}

QIcon ZEditParamLayoutDlg::getIcon(const QStandardItem *pItem) 
{
    int control = pItem->data(ROLE_PARAM_CONTROL).toInt();
    int type = pItem->data(ROLE_ELEMENT_TYPE).toInt();
    if (type == VPARAM_TAB) 
    {
        return QIcon(":/icons/parameter_control_tab.svg");
    } 
    else if (type == VPARAM_GROUP) 
    {
        return QIcon();// ":/icons/parameter_control_group.svg");
    } 
    else if (type != VPARAM_ROOT) 
    {
        if (control == zeno::NullControl)
            return QIcon();
        for (int i = 0; i < sizeof(controlList) / sizeof(CONTROL_ITEM_INFO); i++) 
        {
            if (control == controlList[i].ctrl) 
            {
                return QIcon(controlList[i].icon);
            }
        }
    }
    return QIcon();
}

void ZEditParamLayoutDlg::onComboTableItemsCellChanged(int row, int column)
{
    //dump to item.
    QModelIndex layerIdx = m_ui->paramsView->currentIndex();
    QString value = layerIdx.data(ROLE_PARAM_VALUE).toString();
    if (!layerIdx.isValid() && layerIdx.data(ROLE_VPARAM_TYPE) != VPARAM_PARAM)
        return;

    QStringList lst;
    for (int r = 0; r < m_ui->itemsTable->rowCount(); r++)
    {
        QTableWidgetItem* pItem = m_ui->itemsTable->item(r, 0);
        if (pItem && !pItem->text().isEmpty()) {
            if (lst.contains(pItem->text())) 
            {
                QMessageBox::information(this, tr("Info"), tr("The %1 item already exists").arg(pItem->text()));
                disconnect(m_ui->itemsTable, SIGNAL(cellChanged(int, int)), this, SLOT(onComboTableItemsCellChanged(int, int)));
                pItem->setText("");
                connect(m_ui->itemsTable, SIGNAL(cellChanged(int, int)), this, SLOT(onComboTableItemsCellChanged(int, int)));
                return;
            }
            lst.append(pItem->text());
        }
    }
    if (lst.isEmpty())
        return;

    CONTROL_PROPERTIES properties = layerIdx.data(ROLE_PARAM_CTRL_PROPERTIES).value<CONTROL_PROPERTIES>();
    properties["items"] = lst;

    proxyModelSetData(layerIdx, properties, ROLE_PARAM_CTRL_PROPERTIES);

    if (row == m_ui->itemsTable->rowCount() - 1)
    {
        m_ui->itemsTable->insertRow(m_ui->itemsTable->rowCount());
        m_ui->m_pUpButton->setEnabled(true);
    }

    //update control.
    QLayoutItem *pLayoutItem = m_ui->gridLayout->itemAtPosition(rowValueControl, 1);
    if (pLayoutItem) {
        QComboBox *pControl = qobject_cast<QComboBox *>(pLayoutItem->widget());
        if (pControl) {
            pControl->clear();
            pControl->addItems(lst);
            if (lst.contains(value)) {
                pControl->setCurrentText(value);
            } else {
                proxyModelSetData(layerIdx, lst[0], ROLE_PARAM_VALUE);
            }
        }
    }
}

void ZEditParamLayoutDlg::proxyModelSetData(const QModelIndex& index, const QVariant& newValue, int role)
{
    const QString& objPath = index.data(ROLE_OBJPATH).toString();
    m_paramsLayoutM->setData(index, newValue, role);
}

void ZEditParamLayoutDlg::onParamTreeDeleted()
{
    if (m_ui->itemsTable->hasFocus()) {
        int row = m_ui->itemsTable->currentRow();
        if (row < m_ui->itemsTable->rowCount() - 1) {
            m_ui->itemsTable->removeRow(row);
            onComboTableItemsCellChanged(row, 0);
        }
    } else {
        QModelIndex idx = m_ui->paramsView->currentIndex();
        bool bEditable = true;//TODO:
        if (!idx.isValid() || !idx.parent().isValid() || !bEditable)
            return;

        QString existedName = idx.data(ROLE_MAP_TO_PARAMNAME).toString();
        if (!existedName.isEmpty()) {
            int flag = QMessageBox::question(this, "", "The current item is mapped to a existing param, are you sure to delete it?", QMessageBox::Yes | QMessageBox::No);
            if (flag & QMessageBox::No) {
                return;
            }
        }

        m_paramsLayoutM->removeRow(idx.row(), idx.parent());
    }
}

void ZEditParamLayoutDlg::onTreeCurrentChanged(const QModelIndex& current, const QModelIndex& previous)
{
    auto pCurrentItem = m_paramsLayoutM->itemFromIndex(current);
    if (!pCurrentItem)
        return;

    const QString& name = pCurrentItem->data(ROLE_PARAM_NAME).toString();
    m_ui->editName->setText(name);
    bool bEditable = true;// m_proxyModel->isEditable(current);
    m_ui->editName->setEnabled(bEditable);
    m_ui->editLabel->setText(pCurrentItem->data(ROLE_PARAM_TOOLTIP).toString());

    //delete old control.
    QLayoutItem* pLayoutItem = m_ui->gridLayout->itemAtPosition(rowValueControl, 1);
    if (pLayoutItem)
    {
        QWidget* pControlWidget = pLayoutItem->widget();
        delete pControlWidget;
    }

    VPARAM_TYPE type = (VPARAM_TYPE)pCurrentItem->data(ROLE_ELEMENT_TYPE).toInt();
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
        QStandardItem* parentItem = pCurrentItem->parent();
        zeno::ParamControl ctrl = (zeno::ParamControl)pCurrentItem->data(ROLE_PARAM_CONTROL).toInt();
        const zeno::ParamType paramType = (zeno::ParamType)pCurrentItem->data(ROLE_PARAM_TYPE).toInt();

        const QString& ctrlName = ctrl != zeno::NullControl ? getControl(ctrl, paramType).name : "";
        QVariant controlProperties = pCurrentItem->data(ROLE_PARAM_CTRL_PROPERTIES);
        
        if (true) 
        {
            const QString &parentName = parentItem->data(ROLE_PARAM_NAME).toString();
            //ZASSERT_EXIT(parentName == iotags::params::node_inputs || parentName == iotags::params::node_outputs);
            //bool bInput = parentName == iotags::params::node_inputs;

            QVariant deflVal = pCurrentItem->data(ROLE_PARAM_VALUE);

            CallbackCollection cbSets;
            cbSets.cbEditFinished = [=](QVariant newValue) {
                proxyModelSetData(pCurrentItem->index(), newValue, ROLE_PARAM_VALUE);
            };
            if (!deflVal.isValid())
                deflVal = UiHelper::initDefaultValue(paramType);

            cbSets.cbGetIndexData = [=]() -> QVariant {
                if (!pCurrentItem->data(ROLE_PARAM_VALUE).isValid()) {
                    return UiHelper::initDefaultValue(paramType);
                }
                return pCurrentItem->data(ROLE_PARAM_VALUE);
            };
            QWidget *valueControl = zenoui::createWidget(deflVal, ctrl, paramType, cbSets, controlProperties);
            if (valueControl) {
                valueControl->setEnabled(bEditable);
                m_ui->gridLayout->addWidget(valueControl, rowValueControl, 1);
            }
        }

        {
            BlockSignalScope scope(m_ui->cbControl);

            m_ui->cbControl->setEnabled(true);
            m_ui->cbControl->setCurrentText(ctrlName);
            QString descType = UiHelper::getTypeDesc(paramType);
            if (ctrl == zeno::Seperator) 
            {
                m_ui->cbControl->setEnabled(false);
            }
        }

        /*
        if (pCurrentItem->m_index.isValid()) 
        {
            const QString &refName = pCurrentItem->m_index.data(ROLE_PARAM_NAME).toString();
            m_ui->editCoreParamName->setText(refName);
            const QString &refType = pCurrentItem->m_index.data(ROLE_PARAM_TYPE).toString();
            m_ui->editCoreParamType->setText(refType);
        } 
        else 
        {
            m_ui->editCoreParamName->setText(ctrlName);
            m_ui->editCoreParamType->setText(paramType);
        }
        */

        m_ui->itemsTable->setRowCount(0);

        switchStackProperties(ctrl, pCurrentItem);
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
    VPARAM_TYPE type = (VPARAM_TYPE)layerIdx.data(ROLE_ELEMENT_TYPE).toInt();
    auto pItem = m_paramsLayoutM->itemFromIndex(layerIdx);
    ZASSERT_EXIT(pItem);
    QStringList existNames;
    for (int r = 0; r < pItem->rowCount(); r++)
    {
        QStandardItem* pChildItem = pItem->child(r);
        ZASSERT_EXIT(pChildItem);
        QString _name = pChildItem->data(ROLE_PARAM_NAME).toString();
        existNames.append(_name);
    }

    if (ctrlName == "Tab")
    {
        if (type != VPARAM_ROOT)
        {
            QMessageBox::information(this, tr("Error"), tr("create tab needs to place under the root"));
            return;
        }
        QString newTabName = UiHelper::getUniqueName(existNames, "Tab");
        auto pNewItem = new QStandardItem(newTabName);
        pNewItem->setData(ROLE_ELEMENT_TYPE, VPARAM_TAB);
        pNewItem->setData(getIcon(pNewItem), Qt::DecorationRole);
        pItem->appendRow(pNewItem);
    }
    else if (ctrlName == "Group")
    {
        if (type != VPARAM_TAB)
        {
            QMessageBox::information(this, tr("Error "), tr("create group needs to place under the tab"));
            return;
        }
        QString newGroup = UiHelper::getUniqueName(existNames, "Group");
        auto pNewItem = new QStandardItem(newGroup);
        pNewItem->setData(VPARAM_GROUP, ROLE_ELEMENT_TYPE);
        pNewItem->setData(getIcon(pNewItem), Qt::DecorationRole);
        pItem->appendRow(pNewItem);
    }
    else
    {
        if (type != VPARAM_GROUP)
        {
            QMessageBox::information(this, tr("Error "), tr("create control needs to place under the group"));
            return;
        }
        bool bEditable = true;
        if (!bEditable) {
            QMessageBox::information(this, tr("Error "), tr("The Group cannot be edited"));
            return;
        }
        CONTROL_ITEM_INFO ctrl = getControlByName(ctrlName);
        QString newParamName = UiHelper::getUniqueName(existNames, ctrl.name);
        auto pNewItem = new QStandardItem(newParamName);
        pNewItem->setData(newParamName, ROLE_PARAM_NAME);
        pNewItem->setData(ctrl.ctrl, ROLE_PARAM_CONTROL);
        pNewItem->setData(ctrl.type, ROLE_PARAM_TYPE);
        pNewItem->setData(VPARAM_PARAM, ROLE_ELEMENT_TYPE);
        pNewItem->setData(UiHelper::initDefaultValue(ctrl.type), ROLE_PARAM_VALUE);

        //init properties.
        switch (ctrl.ctrl)
        {
            case zeno::SpinBoxSlider:
            case zeno::SpinBox:
            case zeno::DoubleSpinBox:
            case zeno::Slider:
            {
                CONTROL_PROPERTIES properties;
                properties["step"] = 1;
                properties["min"] = 0;
                properties["max"] = 100;
                pNewItem->setData(properties, ROLE_PARAM_CTRL_PROPERTIES);
                break;
            }
            case zeno::Seperator: 
            {
                //pNewItem->m_sockProp = SOCKPROP_GROUP_LINE;
                break;
            }
        }
        pItem->appendRow(pNewItem);
        pNewItem->setData(getIcon(pNewItem), Qt::DecorationRole);
    }
}

void ZEditParamLayoutDlg::switchStackProperties(int ctrl, QStandardItem* pItem)
{
    QVariant controlProperties = pItem->data(ROLE_PARAM_CTRL_PROPERTIES);
    if (ctrl == zeno::Combobox) {
        CONTROL_PROPERTIES pros = controlProperties.toMap();
        
        if (pros.find("items") != pros.end()) {
                const QStringList &items = pros["items"].toStringList();
                m_ui->itemsTable->setRowCount(items.size() + 1);
                QString value = pItem->data(ROLE_PARAM_VALUE).toString();
                for (int r = 0; r < items.size(); r++) {
                    QTableWidgetItem *newItem = new QTableWidgetItem(items[r]);
                    m_ui->itemsTable->setItem(r, 0, newItem);
                    QLayoutItem *pLayoutItem = m_ui->gridLayout->itemAtPosition(rowValueControl, 1);
                    if (pLayoutItem) {
                        QComboBox *pControl = qobject_cast<QComboBox *>(pLayoutItem->widget());
                        if (pControl) {
                            pControl->setCurrentText(value);
                            proxyModelSetData(pItem->index(), value, ROLE_PARAM_VALUE);
                        }
                    }
                }
        } else {
                m_ui->itemsTable->setRowCount(1);
        }
        m_ui->stackProperties->setCurrentIndex(1);
    } 
    else if (ctrl == zeno::Slider ||
             ctrl == zeno::SpinBox ||
             ctrl == zeno::SpinBoxSlider ||
             ctrl == zeno::DoubleSpinBox)
    {
        m_ui->stackProperties->setCurrentIndex(2);
        QVariantMap map = controlProperties.toMap();
        SLIDER_INFO info;
        if (!map.isEmpty()) {
                info.step = map["step"].toDouble();
                info.min = map["min"].toDouble();
                info.max = map["max"].toDouble();
        } else {
                CONTROL_PROPERTIES properties;
                properties["step"] = info.step;
                properties["min"] = info.min;
                properties["max"] = info.max;
                pItem->setData(properties, ROLE_PARAM_CTRL_PROPERTIES);
        }
        m_ui->editStep->setText(QString::number(info.step));
        m_ui->editMin->setText(QString::number(info.min));
        m_ui->editMax->setText(QString::number(info.max));
    } else {
        m_ui->stackProperties->setCurrentIndex(0);
    }
}

void ZEditParamLayoutDlg::onProxyItemNameChanged(const QModelIndex& itemIdx, const QString& oldPath, const QString& newName)
{
    //ViewParamSetDataCommand *pCmd = new ViewParamSetDataCommand(m_pGraphsModel, oldPath, newName, ROLE_PARAM_NAME);
    //m_commandSeq.append(pCmd);
    if (m_ui->editName->text() != newName)
        m_ui->editName->setText(newName);
}

void ZEditParamLayoutDlg::onViewParamDataChanged(const QModelIndex &topLeft, const QModelIndex &bottomRight, const QVector<int> &roles) 
{
    if (roles.isEmpty())
        return;
    int role = roles[0];
    if (role == ROLE_PARAM_CONTROL) 
    {
        QStandardItem *item = m_paramsLayoutM->itemFromIndex(topLeft);
        QIcon icon = getIcon(item);
        item->setData(icon, Qt::DecorationRole);
    }
}

static QStandardItem* /*VParamItem::*/getItem(QStandardItem* pItem, const QString& uniqueName, int* targetIdx)
{
    for (int r = 0; r < pItem->rowCount(); r++)
    {
        auto pChild = pItem->child(r);
        if (pChild->text() == uniqueName)
        {
            if (targetIdx)
                *targetIdx = r;
            return pChild;
        }
    }
    return nullptr;
}


void ZEditParamLayoutDlg::onNameEditFinished()
{
    const QModelIndex& currIdx = m_ui->paramsView->currentIndex();
    if (!currIdx.isValid())
        return;

    QStandardItem* pItem = m_paramsLayoutM->itemFromIndex(currIdx);
    QString oldPath = pItem->data(ROLE_OBJPATH).toString();
    QString newName = m_ui->editName->text();
    QString oldName = pItem->data(ROLE_PARAM_NAME).toString();
    if (oldName != newName) {
        int dstRow = 0;
        QStandardItem* pTargetGroup = pItem->parent();
        if (pTargetGroup && !getItem(pTargetGroup, newName, &dstRow))
        {
            pItem->setData(newName, ROLE_PARAM_NAME);
            pItem->setText(newName);
            //onProxyItemNameChanged(pItem->index(), oldPath, newName);
        } 
        else 
        {
            disconnect(m_ui->editName, SIGNAL(editingFinished()), this, SLOT(onNameEditFinished()));
            m_ui->editName->setText(oldName);
            connect(m_ui->editName, SIGNAL(editingFinished()), this, SLOT(onNameEditFinished()));
            QMessageBox::information(this, tr("Info"), tr("The param name already exists"));
        }
    }
}

void ZEditParamLayoutDlg::onLabelEditFinished()
{
    const QModelIndex &currIdx = m_ui->paramsView->currentIndex();
    if (!currIdx.isValid())
        return;

    QStandardItem *pItem = m_paramsLayoutM->itemFromIndex(currIdx);

    ZASSERT_EXIT(pItem);
    QString oldText = pItem->data(ROLE_PARAM_NAME).toString();
    QString newText = m_ui->editLabel->text();
    if (oldText == newText)
        return;
    pItem->setData(newText, ROLE_PARAM_TOOLTIP);
}

void ZEditParamLayoutDlg::onHintEditFinished()
{

}

void ZEditParamLayoutDlg::onMinEditFinished()
{
    QModelIndex layerIdx = m_ui->paramsView->currentIndex();
    if (!layerIdx.isValid() && layerIdx.data(ROLE_VPARAM_TYPE) != VPARAM_PARAM)
        return;

    CONTROL_PROPERTIES properties = layerIdx.data(ROLE_PARAM_CTRL_PROPERTIES).value<CONTROL_PROPERTIES>();
    qreal from = m_ui->editMin->text().toDouble();
    properties["min"] = from;
    proxyModelSetData(layerIdx, properties, ROLE_PARAM_CTRL_PROPERTIES);
    updateSliderInfo();
}

void ZEditParamLayoutDlg::onMaxEditFinished()
{
    QModelIndex layerIdx = m_ui->paramsView->currentIndex();
    if (!layerIdx.isValid() && layerIdx.data(ROLE_VPARAM_TYPE) != VPARAM_PARAM)
        return;

    CONTROL_PROPERTIES properties = layerIdx.data(ROLE_PARAM_CTRL_PROPERTIES).value<CONTROL_PROPERTIES>();
    qreal to = m_ui->editMax->text().toDouble();
    properties["max"] = to;
    proxyModelSetData(layerIdx, properties, ROLE_PARAM_CTRL_PROPERTIES);
    updateSliderInfo();
}

void ZEditParamLayoutDlg::onControlItemChanged(int idx)
{
    const QString& controlName = m_ui->cbControl->itemText(idx);
    zeno::ParamControl ctrl = UiHelper::getControlByDesc(controlName);
    QModelIndex layerIdx = m_ui->paramsView->currentIndex();
    if (!layerIdx.isValid() && layerIdx.data(ROLE_VPARAM_TYPE) != VPARAM_PARAM)
        return;

    proxyModelSetData(layerIdx, ctrl, ROLE_PARAM_CONTROL);

    QLayoutItem* pLayoutItem = m_ui->gridLayout->itemAtPosition(rowValueControl, 1);
    if (pLayoutItem)
    {
        QWidget* pControlWidget = pLayoutItem->widget();
        delete pControlWidget;
    }

    CallbackCollection cbSets;
    cbSets.cbEditFinished = [=](QVariant newValue) {
        proxyModelSetData(layerIdx, newValue, ROLE_PARAM_VALUE);
    };

    zeno::ParamType type = getTypeByControlName(controlName);

    //update type:
    auto pItem = m_paramsLayoutM->itemFromIndex(layerIdx);
    pItem->setData(type, ROLE_PARAM_TYPE);

    QVariant value = layerIdx.data(ROLE_PARAM_VALUE);
    if (!value.isValid())
        value = UiHelper::initDefaultValue(type);

    QVariant controlProperties = layerIdx.data(ROLE_PARAM_CTRL_PROPERTIES);
    cbSets.cbGetIndexData = [=]() -> QVariant { return UiHelper::initDefaultValue(type); };
    QWidget *valueControl = zenoui::createWidget(value, ctrl, type, cbSets, controlProperties);
    if (valueControl) {
        valueControl->setEnabled(true);
        m_ui->gridLayout->addWidget(valueControl, rowValueControl, 1);
        
        switchStackProperties(ctrl, pItem);
        pItem->setData(getIcon(pItem), Qt::DecorationRole);
    }
}

void ZEditParamLayoutDlg::onStepEditFinished()
{
    QModelIndex layerIdx = m_ui->paramsView->currentIndex();
    if (!layerIdx.isValid() && layerIdx.data(ROLE_VPARAM_TYPE) != VPARAM_PARAM)
        return;

    CONTROL_PROPERTIES properties = layerIdx.data(ROLE_PARAM_CTRL_PROPERTIES).value<CONTROL_PROPERTIES>();
    qreal step = m_ui->editStep->text().toDouble();
    properties["step"] = step;

    const QString &objPath = layerIdx.data(ROLE_OBJPATH).toString();
    //ViewParamSetDataCommand *pCommand = new ViewParamSetDataCommand(m_pGraphsModel, objPath, properties, ROLE_PARAM_CTRL_PROPERTIES);
    //m_commandSeq.append(pCommand);

    m_paramsLayoutM->setData(layerIdx, properties, ROLE_PARAM_CTRL_PROPERTIES);
    updateSliderInfo();
}

void ZEditParamLayoutDlg::onChooseParamClicked()
{
    //TODO OR NOT TO DO: map to core param
    /*
    ZMapCoreparamDlg dlg(m_nodeIdx);
    if (QDialog::Accepted == dlg.exec())
    {
        QModelIndex coreIdx = dlg.coreIndex();

        QModelIndex viewparamIdx = m_ui->paramsView->currentIndex();
        QStandardItem* pItem = m_paramsLayoutM->itemFromIndex(viewparamIdx);
        VParamItem* pViewItem = static_cast<VParamItem*>(pItem);

        if (coreIdx.isValid())
        {
            pViewItem->mapCoreParam(coreIdx);

            //update control info.
            const QString& newName = coreIdx.data(ROLE_PARAM_NAME).toString();
            const QString& typeDesc = coreIdx.data(ROLE_PARAM_TYPE).toString();

            m_ui->editCoreParamName->setText(newName);
            m_ui->editCoreParamType->setText(typeDesc);

            PARAM_CONTROL newCtrl = UiHelper::getControlByType(typeDesc);
            pViewItem->setData(newCtrl, ROLE_PARAM_CONTROL);
        }
        else
        {
            pViewItem->m_index = QModelIndex();
            pViewItem->setData(zeno::NullControl, ROLE_PARAM_CONTROL);
            m_ui->editCoreParamName->setText("");
            m_ui->editCoreParamType->setText("");
        }
    }
    */
}

void ZEditParamLayoutDlg::applyForItem(QStandardItem* proxyItem, QStandardItem* appliedItem)
{
    //oldItem: proxy.
#if 0
    QString subgName;
    QModelIndex subgIdx;
    bool bSubInput = false;
    bool bApplySubnetParam = m_bSubgraphNode && proxyItem->data(ROLE_VPARAM_TYPE) == VPARAM_GROUP;
    if (bApplySubnetParam)
    {
        VParamItem *pGroup = static_cast<VParamItem *>(proxyItem);
        bSubInput = pGroup->m_name == iotags::params::node_inputs;
        subgName = m_nodeIdx.data(ROLE_CLASS_NAME).toString();
        subgIdx = m_pGraphsModel->index(subgName);
    }

    int r = 0;
    for (r = 0; r < proxyItem->rowCount(); r++)
    {
        VParamItem* pCurrent = static_cast<VParamItem*>(proxyItem->child(r));
        uint uuid = pCurrent->m_uuid;
        const QString name = pCurrent->data(ROLE_PARAM_NAME).toString();
        const QString typeDesc = pCurrent->data(ROLE_PARAM_TYPE).toString();
        const QVariant value = pCurrent->data(ROLE_PARAM_VALUE);
        const PARAM_CONTROL ctrl = (PARAM_CONTROL)pCurrent->data(ROLE_PARAM_CONTROL).toInt();
        QVariant ctrlProperties;
        if (pCurrent->m_customData.find(ROLE_PARAM_CTRL_PROPERTIES) != pCurrent->m_customData.end()) 
		{
            ctrlProperties = pCurrent->m_customData[ROLE_PARAM_CTRL_PROPERTIES];
        }

        int targetRow = 0;
        VParamItem* pTarget = nullptr;
        for (int _r = 0; _r < appliedItem->rowCount(); _r++)
        {
            VParamItem* pChild = static_cast<VParamItem*>(appliedItem->child(_r));
            if (pChild->m_uuid == uuid)
            {
                pTarget = pChild;
                targetRow = _r;
                break;
            }
        }

        if (pTarget)
        {
            if (targetRow != r)
            {
                //move first.
                const int srcRow = targetRow;
                const int dstRow = r;

                //update desc.
                NODE_DESC desc;
                bool ret = m_pGraphsModel->getDescriptor(subgName, desc);
                if (bApplySubnetParam)
                {
                    if (bSubInput) {
                        int sz = desc.inputs.size();
                        if (sz > srcRow && sz > dstRow) {
                            desc.inputs.move(srcRow, dstRow);
                            m_pGraphsModel->updateSubgDesc(subgName, desc);
                        }
                    }
                    else {
                        int sz = desc.outputs.size();
                        if (sz > srcRow && sz > dstRow) {
                            desc.outputs.move(srcRow, dstRow);
                            m_pGraphsModel->updateSubgDesc(subgName, desc);
                        }
                    }
                }

                //update the corresponding order for every subgraph node.
                QModelIndexList subgNodes = m_pGraphsModel->findSubgraphNode(subgName);
                for (auto idx : subgNodes)
                {
                    NodeParamModel* nodeParams = QVariantPtr<NodeParamModel>::asPtr(idx.data(ROLE_NODE_PARAMS));
                    ZASSERT_EXIT(nodeParams);
                    VParamItem* pGroup = bSubInput ? nodeParams->getInputs() : nodeParams->getOutputs();
                    ZASSERT_EXIT(pGroup);
                    QModelIndex parent = pGroup->index();
                    nodeParams->moveRow(parent, srcRow, parent, dstRow);
                }
                if (qobject_cast<NodeParamModel*>(m_model) == nullptr)
                {
                    PanelParamModel *panelParams = QVariantPtr<PanelParamModel>::asPtr(m_model->nodeIdx().data(ROLE_PANEL_PARAMS));
                    ZASSERT_EXIT(panelParams);
                    QModelIndex parent = pTarget->parent()->index();
                    panelParams->moveRow(parent, srcRow, parent, dstRow);
                }
               //reacquire pTarget, because the implementation of moveRow is simplily
                pTarget = static_cast<VParamItem *>(appliedItem->child(r));
            }

            //the corresponding item exists.
            if (name != pTarget->m_name)
            {
                //rename
                QString oldName = pTarget->m_name;
                QString newName = name;
                if (bApplySubnetParam)
                {
                    if (ctrl == CONTROL_GROUP_LINE) 
                    {
                        updateControlGroup(bSubInput, newName, oldName, ctrl, r);
                    } 
                    else 
                    {
                        //get subinput name idx, update its value, and then sync to all subgraph node.
                        const QModelIndex &subInOutput = UiHelper::findSubInOutputIdx(m_pGraphsModel, bSubInput, oldName, subgIdx);
                        NodeParamModel *nodeParams = QVariantPtr<NodeParamModel>::asPtr(subInOutput.data(ROLE_NODE_PARAMS));
                        ZASSERT_EXIT(nodeParams);
                        const QModelIndex &nameIdx = nodeParams->getParam(PARAM_PARAM, "name");
                        //update the value on "name" in SubInput/SubOutput.
                        m_pGraphsModel->ModelSetData(nameIdx, newName, ROLE_PARAM_VALUE);
                    }
                }
                else
                {
                    pTarget->setData(newName, ROLE_PARAM_NAME);
                }
            }
            if (ctrl == CONTROL_GROUP_LINE)
                continue;
            if (pCurrent->vType == VPARAM_PARAM)
            {
                //check type
                if (pTarget->data(ROLE_PARAM_TYPE) != typeDesc)
                {
                    if (bApplySubnetParam) {
                        //get subinput type idx, update its value, and then sync to all subgraph node.
                        const QModelIndex &subInOutput =
                         UiHelper::findSubInOutputIdx(m_pGraphsModel, bSubInput, pTarget->m_name, subgIdx);
                        NodeParamModel *nodeParams = QVariantPtr<NodeParamModel>::asPtr(subInOutput.data(ROLE_NODE_PARAMS));
                        ZASSERT_EXIT(nodeParams);
                        const QModelIndex &typelIdx = nodeParams->getParam(PARAM_PARAM, "type");
                        m_pGraphsModel->ModelSetData(typelIdx, typeDesc, ROLE_PARAM_VALUE);
                    } else {
                        pTarget->setData(typeDesc, ROLE_PARAM_TYPE);
                    }
                }

                //check default value
                if (pTarget->data(ROLE_PARAM_VALUE) != value)
                {
                    if (bApplySubnetParam) {
                        //get subinput defl idx, update its value, and then sync to all subgraph node.
                        const QModelIndex &subInOutput = UiHelper::findSubInOutputIdx(m_pGraphsModel, bSubInput, pTarget->m_name, subgIdx);
                        NodeParamModel *nodeParams = QVariantPtr<NodeParamModel>::asPtr(subInOutput.data(ROLE_NODE_PARAMS));
                        ZASSERT_EXIT(nodeParams);
                        const QModelIndex &deflIdx = nodeParams->getParam(PARAM_PARAM, "defl");
                        //update the value on "defl" in SubInput/SubOutput.
                        m_pGraphsModel->ModelSetData(deflIdx, value, ROLE_PARAM_VALUE);
                    } else {
                        pTarget->setData(value, ROLE_PARAM_VALUE);
                    }
                }

                //control
                if (pTarget->m_ctrl != ctrl)
                {
                    if (bApplySubnetParam) {
                        //get subinput defl idx, update its value, and then sync to all subgraph node.
                        const QModelIndex &subInOutput = UiHelper::findSubInOutputIdx(m_pGraphsModel, bSubInput, pTarget->m_name, subgIdx);
                        NodeParamModel *nodeParams = QVariantPtr<NodeParamModel>::asPtr(subInOutput.data(ROLE_NODE_PARAMS));
                        ZASSERT_EXIT(nodeParams);
                        const QModelIndex &deflIdx = nodeParams->getParam(PARAM_PARAM, "defl");
                        //update the control on "defl" in SubInput/SubOutput.
                        m_pGraphsModel->ModelSetData(deflIdx, ctrl, ROLE_PARAM_CONTROL);
                    } 
                    pTarget->setData(ctrl, ROLE_PARAM_CONTROL);
                }

	            //control properties
                bool isChanged = pTarget->m_customData[ROLE_PARAM_CTRL_PROPERTIES] != ctrlProperties;
                if (isChanged) 
                {
                    if (bApplySubnetParam)
                    {
                        //get subinput defl idx, update its value, and then sync to all subgraph node.
                        const QModelIndex& subInOutput = UiHelper::findSubInOutputIdx(m_pGraphsModel, bSubInput, pTarget->m_name, subgIdx);
                        if (subInOutput.isValid())
                        {
                            NodeParamModel* nodeParams = QVariantPtr<NodeParamModel>::asPtr(subInOutput.data(ROLE_NODE_PARAMS));
                            ZASSERT_EXIT(nodeParams);
                            const QModelIndex &deflIdx = nodeParams->getParam(PARAM_PARAM, "defl");
                            //update the value properties on "defl" in SubInput/SubOutput.
                            m_pGraphsModel->ModelSetData(deflIdx, ctrlProperties, ROLE_PARAM_CTRL_PROPERTIES);
                        }
                    }
                    pTarget->setData(ctrlProperties, ROLE_PARAM_CTRL_PROPERTIES);
                }
                //core mapping
                if (pCurrent->m_index != pTarget->m_index) 
                {
                    pTarget->mapCoreParam(pCurrent->m_index);
                }
                //tool tip
                QVariant newTip;
                if (pCurrent->m_customData.find(ROLE_PARAM_TOOLTIP) != pCurrent->m_customData.end()) {
                    newTip = pCurrent->m_customData[ROLE_PARAM_TOOLTIP];
                }
                QVariant oldTip = pTarget->m_customData[ROLE_PARAM_TOOLTIP];
                if (oldTip != newTip) 
                {
                    pTarget->setData(newTip, ROLE_PARAM_TOOLTIP);
                }
            }
            else
            {
                applyForItem(pCurrent, pTarget);
            }
        }
        else
        {
            //new item.
            if (pCurrent->vType == VPARAM_PARAM)
            {
                if (m_bSubgraphNode) 
                {
                    int srcRow = 0;
                    if (ctrl == CONTROL_GROUP_LINE) 
                    {
                        addControlGroup(bSubInput, name, ctrl);
                        QStandardItem *pNewItem = pCurrent->clone();
                        appliedItem->appendRow(pNewItem);
                        srcRow = appliedItem->rowCount() - 1;
                    } 
                    else 
                    {
                        const QVariant &defl = pCurrent->data(ROLE_PARAM_VALUE);
                        const QString &typeDesc = pCurrent->data(ROLE_PARAM_TYPE).toString();
                        const PARAM_CONTROL ctrl = pCurrent->m_ctrl;

                        QPointF pos(0, 0);
                        //todo: node arrangement.
                        VParamItem *pGroup = static_cast<VParamItem *>(proxyItem);
                        VParamItem *pTargetGroup = static_cast<VParamItem *>(appliedItem);

                        NODE_DATA node =
                            NodesMgr::newNodeData(m_pGraphsModel, bSubInput ? "SubInput" : "SubOutput", pos);
                        PARAMS_INFO params = node[ROLE_PARAMETERS].value<PARAMS_INFO>();
                        params["name"].value = name;
                        params["name"].toolTip = m_ui->editLabel->text();
                        params["type"].value = typeDesc;
                        params["defl"].typeDesc = typeDesc;
                        params["defl"].value = defl;
                        params["defl"].control = ctrl;
                        params["defl"].controlProps = pCurrent->data(ROLE_PARAM_CTRL_PROPERTIES);
                        node[ROLE_PARAMETERS] = QVariant::fromValue(params);

                        m_pGraphsModel->addNode(node, subgIdx, true);

                        //the newItem is created just now, after adding the subgraph node.
                        
                        VParamItem* newItem = pTargetGroup->getItem(name, &srcRow);
                        ZASSERT_EXIT(newItem);
                        pCurrent->m_uuid = newItem->m_uuid;
                    }

                    if (srcRow != r)
                    {
                        //update desc.
                        NODE_DESC desc;
                        bool ret = m_pGraphsModel->getDescriptor(subgName, desc);
                        if (bSubInput) {
                            int sz = desc.inputs.size();
                            if (sz > r && sz > srcRow) {
                                desc.inputs.move(srcRow, r);
                                m_pGraphsModel->updateSubgDesc(subgName, desc);
                            }
                        }
                        else {
                            int sz = desc.outputs.size();
                            if (sz > r && sz > srcRow) {
                                desc.outputs.move(srcRow, r);
                                m_pGraphsModel->updateSubgDesc(subgName, desc);
                            }
                        }

                        //update the corresponding order for every subgraph node.
                        QModelIndexList subgNodes = m_pGraphsModel->findSubgraphNode(subgName);
                        for (const auto& idx : subgNodes)
                        {
                            NodeParamModel* nodeParams = QVariantPtr<NodeParamModel>::asPtr(idx.data(ROLE_NODE_PARAMS));
                            ZASSERT_EXIT(nodeParams);
                            VParamItem* pGroup = bSubInput ? nodeParams->getInputs() : nodeParams->getOutputs();
                            ZASSERT_EXIT(pGroup);
                            QModelIndex parent = pGroup->index();
                            nodeParams->moveRow(parent, srcRow, parent, r);
                        }
                    }

                } 
                else 
                {
                    QStandardItem *pNewItem = pCurrent->clone();
                    appliedItem->appendRow(pNewItem);
                }
            }
            else
            {
                QStandardItem* pNewItem = pCurrent->clone();
                appliedItem->appendRow(pNewItem);
            }
        }
    }

    //remove items which don't exist in proxyitem.
    QStringList deleteParams;
    for (int _r = 0; _r < appliedItem->rowCount(); _r++)
    {
        VParamItem* pExisted = static_cast<VParamItem*>(appliedItem->child(_r));
        bool bDeleted = true;
        for (int r = 0; r < proxyItem->rowCount(); r++)
        {
            VParamItem* pItem = static_cast<VParamItem*>(proxyItem->child(r));
            if (pItem->m_uuid == pExisted->m_uuid)
            {
                bDeleted = false;
                break;
            }
        }
        if (bDeleted)
        {
            deleteParams.append(pExisted->m_name);
        }
    }

    for (QString deleteParam : deleteParams)
    {
        for (int r = 0; r < appliedItem->rowCount(); r++) 
        {
            VParamItem *pItem = static_cast<VParamItem *>(appliedItem->child(r));
            if (pItem->m_name == deleteParam) 
            {
                if (bApplySubnetParam) 
                {
                    if (pItem->m_ctrl == CONTROL_GROUP_LINE) 
                    {
                        delControlGroup(bSubInput, pItem->m_name);
                        appliedItem->removeRow(r);
                    } 
                    else 
                    {
                        const QModelIndex &subInOutput = UiHelper::findSubInOutputIdx(m_pGraphsModel, bSubInput, deleteParam, subgIdx);
                        const QString ident = subInOutput.data(ROLE_NODE_NAME).toString();

                        //currently, the param about subgraph is construct or deconstruct by SubInput/SubOutput node.
                        //so, we have to remove or add the SubInput/SubOutput node to affect the params.
                        m_pGraphsModel->removeNode(ident, subgIdx, true);
                    }
                } 
                else 
                {
                    appliedItem->removeRow(r);
                }
                break;
            }
        }
       
    }
#endif
}

void ZEditParamLayoutDlg::onApply()
{
    //temp case: for only inputs and outputs.
    std::vector<std::pair<zeno::ParamInfo, std::string>> params;

    QStandardItem* pRoot = m_paramsLayoutM->item(0);
    QStandardItem* pInputs = pRoot->child(0);
    for (int i = 0; i < pInputs->rowCount(); i++)
    {
        QStandardItem* pItem = pInputs->child(i);
        zeno::ParamInfo param;

        param.control = (zeno::ParamControl)pItem->data(ROLE_PARAM_CONTROL).toInt();
        param.type = (zeno::ParamType)pItem->data(ROLE_PARAM_TYPE).toInt();
        param.defl = UiHelper::qvarToZVar(pItem->data(ROLE_PARAM_VALUE), param.type);
        param.name = pItem->data(ROLE_PARAM_NAME).toString().toStdString();
        param.tooltip = pItem->data(ROLE_PARAM_TOOLTIP).toString().toStdString();
        const QString& existName = pItem->data(ROLE_MAP_TO_PARAMNAME).toString();

        params.push_back({ param, existName.toStdString() });
    }
    QStandardItem* pOutputs = pRoot->child(1);
    for (int i = 0; i < pOutputs->rowCount(); i++)
    {
        QStandardItem* pItem = pOutputs->child(i);
        zeno::ParamInfo param;

        param.control = (zeno::ParamControl)pItem->data(ROLE_PARAM_CONTROL).toInt();
        param.type = (zeno::ParamType)pItem->data(ROLE_PARAM_TYPE).toInt();
        param.defl = UiHelper::qvarToZVar(pItem->data(ROLE_PARAM_VALUE), param.type);
        param.name = pItem->data(ROLE_PARAM_NAME).toString().toStdString();
        param.tooltip = pItem->data(ROLE_PARAM_TOOLTIP).toString().toStdString();
        const QString& existName = pItem->data(ROLE_MAP_TO_PARAMNAME).toString();

        params.push_back({ param, existName.toStdString() });
    }

    m_model->batchModifyParams(params);
    /*
    m_pGraphsModel->beginTransaction("edit custom param for node");
    zeno::scope_exit scope([=]() { m_pGraphsModel->endTransaction(); });
    applyForItem(m_proxyModel->invisibleRootItem(), m_model->invisibleRootItem());
    */
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

void ZEditParamLayoutDlg::updateSliderInfo()
{
    SLIDER_INFO info;
    info.step = m_ui->editStep->text().toDouble();
    info.min = m_ui->editMin->text().toDouble();
    info.max = m_ui->editMax->text().toDouble();
    //update control.
    QLayoutItem* pLayoutItem = m_ui->gridLayout->itemAtPosition(rowValueControl, 1);
    if (pLayoutItem) {
        if (QDoubleSpinBox* pControl = qobject_cast<QDoubleSpinBox*>(pLayoutItem->widget()))
        {
            pControl->setRange(info.min, info.max);
            pControl->setSingleStep(info.step);
        }
        else if (ZSpinBoxSlider* pControl = qobject_cast<ZSpinBoxSlider*>(pLayoutItem->widget()))
        {
            pControl->setRange(info.min, info.max);
            pControl->setSingleStep(info.step);
        }
        else if (QSpinBox* pControl = qobject_cast<QSpinBox*>(pLayoutItem->widget()))
        {
            pControl->setRange(info.min, info.max);
            pControl->setSingleStep(info.step);
        }
        else if (QSlider* pControl = qobject_cast<QSlider*>(pLayoutItem->widget()))
        {
            pControl->setRange(info.min, info.max);
            pControl->setSingleStep(info.step);
        }
    }
}
