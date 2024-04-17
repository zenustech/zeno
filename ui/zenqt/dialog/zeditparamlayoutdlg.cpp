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
    {"read path",           zeno::ReadPathEdit, zeno::Param_String, ":/icons/parameter_control_fold.svg"},
    {"write path",          zeno::WritePathEdit,zeno::Param_String, ":/icons/parameter_control_fold.svg"},
    {"directory",       zeno::DirectoryPathEdit,zeno::Param_String, ":/icons/parameter_control_fold.svg"},
    {"Enum",                zeno::Combobox,     zeno::Param_String, ":/icons/parameter_control_enum.svg"},
    {"Float Vector 4",      zeno::Vec4edit,     zeno::Param_Vec4f,  ":/icons/parameter_control_floatVector4.svg"},
    {"Float Vector 3",      zeno::Vec3edit,     zeno::Param_Vec3f,  ":/icons/parameter_control_floatVector3.svg"},
    {"Float Vector 2",      zeno::Vec2edit,     zeno::Param_Vec2f,  ":/icons/parameter_control_floatVector2.svg"},
    {"Integer Vector 4",    zeno::Vec4edit,     zeno::Param_Vec4i,  ":/icons/parameter_control_integerVector4.svg"},
    {"Integer Vector 3",    zeno::Vec3edit,     zeno::Param_Vec3i,  ":/icons/parameter_control_integerVector3.svg"},
    {"Integer Vector 2",    zeno::Vec2edit,     zeno::Param_Vec2i,  ":/icons/parameter_control_integerVector2.svg"},
    {"Color",               zeno::Heatmap,      zeno::Param_Heatmap,":/icons/parameter_control_color.svg"},
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



ParamTreeItemDelegate::ParamTreeItemDelegate(QStandardItemModel* model, QObject *parent)
    : QStyledItemDelegate(parent)
    , m_model(model)
{
}

ParamTreeItemDelegate::~ParamTreeItemDelegate()
{
}

QWidget* ParamTreeItemDelegate::createEditor(QWidget* parent, const QStyleOptionViewItem& option, const QModelIndex& index) const
{
    auto pItem = m_model->itemFromIndex(index);
    if (!pItem)
        return nullptr;

    bool bEditable = pItem->isEditable();
    if (!bEditable)
        return nullptr;
    return QStyledItemDelegate::createEditor(parent, option, index);
}

void ParamTreeItemDelegate::setModelData(QWidget *editor, QAbstractItemModel *model, const QModelIndex &index) const 
{
    QString oldName = index.data().toString();
    QString newName = editor->property(editor->metaObject()->userProperty().name()).toString();
    if (oldName != newName) {
        if (m_isGlobalUniqueFunc(newName)) {
            QStyledItemDelegate::setModelData(editor, model, index);
            model->setData(index, newName, ROLE_PARAM_NAME);
        }
        else {
            QMessageBox::information(nullptr, tr("Info"), tr("The param name already exists"));
        }
    }
}

void ParamTreeItemDelegate::paint(QPainter *painter, const QStyleOptionViewItem &option,
                                  const QModelIndex &index) const {
    QStyledItemDelegate::paint(painter, option, index);
}

outputListItemDelegate::outputListItemDelegate(QStandardItemModel* model, QObject* parent)
    : QStyledItemDelegate(parent)
    , m_model(model)
{
}

outputListItemDelegate::~outputListItemDelegate()
{
}

QWidget* outputListItemDelegate::createEditor(QWidget* parent, const QStyleOptionViewItem& option, const QModelIndex& index) const
{
    auto pItem = m_model->itemFromIndex(index);
    if (!pItem)
        return nullptr;

    bool bEditable = pItem->isEditable();
    if (!bEditable)
        return nullptr;
    return QStyledItemDelegate::createEditor(parent, option, index);
}

void outputListItemDelegate::setModelData(QWidget* editor, QAbstractItemModel* model, const QModelIndex& index) const
{
    QString oldName = index.data().toString();
    QString newName = editor->property(editor->metaObject()->userProperty().name()).toString();
    if (oldName != newName) {
        if (m_isGlobalUniqueFunc(newName)) {
            QStyledItemDelegate::setModelData(editor, model, index);
            model->setData(index, newName, ROLE_PARAM_NAME);
        }
        else {
            QMessageBox::information(nullptr, tr("Info"), tr("The param name already exists"));
        }
    }
}

void outputListItemDelegate::paint(QPainter* painter, const QStyleOptionViewItem& option,
    const QModelIndex& index) const {
    QStyledItemDelegate::paint(painter, option, index);
}


ZEditParamLayoutDlg::ZEditParamLayoutDlg(QStandardItemModel* pModel, QWidget* parent)
    : QDialog(parent)
    //, m_paramsLayoutM(pModel)
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
    initModel(pModel);
    initIcon(m_paramsLayoutM_inputs->invisibleRootItem());

    m_ui->paramsView->setModel(m_paramsLayoutM_inputs);
    m_ui->outputsView->setModel(m_paramsLayoutM_outputs);
    ParamTreeItemDelegate* treeDelegate = new ParamTreeItemDelegate(m_paramsLayoutM_inputs, m_ui->paramsView);
    outputListItemDelegate* listDelegate = new outputListItemDelegate(m_paramsLayoutM_outputs, m_ui->outputsView);
    m_ui->paramsView->setItemDelegate(treeDelegate);
    m_ui->outputsView->setItemDelegate(listDelegate);

    m_isGlobalUniqueFunc = [&](QString name) -> bool {
        QStandardItem* pParamsViewRoot = m_paramsLayoutM_inputs->item(0);
        auto paramsResLst = m_paramsLayoutM_inputs->match(pParamsViewRoot->index(), ROLE_PARAM_NAME, name, 1, Qt::MatchRecursive);
        auto outputResLst = m_paramsLayoutM_outputs->match(m_paramsLayoutM_outputs->index(0, 0), ROLE_PARAM_NAME, name, 1, Qt::MatchRecursive);
        return paramsResLst.empty() && outputResLst.empty();
    };
    treeDelegate->m_isGlobalUniqueFunc = m_isGlobalUniqueFunc;
    listDelegate->m_isGlobalUniqueFunc = m_isGlobalUniqueFunc;

    QItemSelectionModel* selModel = m_ui->paramsView->selectionModel();
    connect(selModel, SIGNAL(currentChanged(const QModelIndex &, const QModelIndex &)), this,
            SLOT(onTreeCurrentChanged(const QModelIndex &, const QModelIndex &)));
    QModelIndex selIdx = selModel->currentIndex();
    const QModelIndex& wtfIdx = m_paramsLayoutM_inputs->index(0, 0);
    selModel->setCurrentIndex(wtfIdx, QItemSelectionModel::SelectCurrent);
    m_ui->paramsView->expandAll();

    QItemSelectionModel* selModelOutputs = m_ui->outputsView->selectionModel();
    connect(selModelOutputs, SIGNAL(currentChanged(const QModelIndex&, const QModelIndex&)), this,
        SLOT(onOutputsListCurrentChanged(const QModelIndex&, const QModelIndex&)));
    const QModelIndex& outputFirst = m_paramsLayoutM_outputs->index(0, 0);
    selModelOutputs->setCurrentIndex(outputFirst, QItemSelectionModel::SelectCurrent);

    connect(m_ui->editName, SIGNAL(editingFinished()), this, SLOT(onNameEditFinished()));
    connect(m_ui->editLabel, SIGNAL(editingFinished()), this, SLOT(onLabelEditFinished()));
    connect(m_ui->btnAddInput, SIGNAL(clicked()), this, SLOT(onBtnAddInputs()));
    connect(m_ui->btnAddOutput, SIGNAL(clicked()), this, SLOT(onBtnAddOutputs()));
    connect(m_ui->btnApply, SIGNAL(clicked()), this, SLOT(onApply()));
    connect(m_ui->btnOk, SIGNAL(clicked()), this, SLOT(onOk()));
    connect(m_ui->btnCancel, SIGNAL(clicked()), this, SLOT(onCancel()));

    //QShortcut* shortcut = new QShortcut(QKeySequence(Qt::Key_Delete), m_ui->paramsView);
    //connect(shortcut, SIGNAL(activated()), this, SLOT(onParamTreeDeleted()));
    m_ui->paramsView->installEventFilter(this);
    m_ui->outputsView->installEventFilter(this);

    connect(m_ui->cbControl, SIGNAL(currentIndexChanged(int)), this, SLOT(onControlItemChanged(int)));
    connect(m_ui->cbSocketType, SIGNAL(currentIndexChanged(int)), this, SLOT(onSocketTypeChanged(int)));

    connect(m_paramsLayoutM_inputs, &QStandardItemModel::dataChanged, this, &ZEditParamLayoutDlg::onParamsViewParamDataChanged);
    connect(m_paramsLayoutM_outputs, &QStandardItemModel::dataChanged, this, &ZEditParamLayoutDlg::onOutputsViewParamDataChanged);
    connect(m_ui->paramsView, &QTreeView::clicked, this, [&]() {m_ui->outputsView->selectionModel()->clearCurrentIndex(); });
    connect(m_ui->outputsView, &QListView::clicked, this, [&]() {m_ui->paramsView->selectionModel()->clearCurrentIndex(); });
}

void ZEditParamLayoutDlg::initModel(const QStandardItemModel* pModel)
{
    m_paramsLayoutM_inputs = new QStandardItemModel(this);
    m_paramsLayoutM_outputs = new QStandardItemModel(this);
    auto cloneItem = [](auto const& cloneItem, QStandardItem* pItem)->QStandardItem* {
        QStandardItem* newItem = pItem->clone();
        for (int i = 0; i < pItem->rowCount(); i++)
        {
            QStandardItem* childItem = pItem->child(i);
            newItem->appendRow(cloneItem(cloneItem, childItem));
        }
        return newItem;
    };
    QStandardItem* inputsItem = pModel->item(0, 0);
    m_paramsLayoutM_inputs->appendRow(cloneItem(cloneItem, inputsItem));
    //for (int r = 0; r < pModel->rowCount(); r++)
    //{
    //    QStandardItem* newItem = pModel->item(1, 0);
    //    m_paramsLayoutM_inputs->appendRow(cloneItem(cloneItem, newItem));
    //}
}

void ZEditParamLayoutDlg::initUI() 
{
    m_ui->editHint->setProperty("cssClass", "dark");
    m_ui->editName->setProperty("cssClass", "dark");
    m_ui->editLabel->setProperty("cssClass", "dark");
    m_ui->splitter->setStyleSheet("QSplitter::handle {background-color: rgb(31,31,31);}");
    m_ui->labelCreateControl->setProperty("cssClass", "bold");
    m_ui->labelPrameter->setProperty("cssClass", "bold");
    m_ui->labelSetting->setProperty("cssClass", "bold");
    m_ui->listConctrl->setFixedWidth(ZenoStyle::dpiScaled(296));
    m_ui->paramsView->setFixedWidth(ZenoStyle::dpiScaled(296));
    m_ui->outputsView->setFixedWidth(ZenoStyle::dpiScaled(296));
    m_ui->btnAddInput->setFixedSize(ZenoStyle::dpiScaled(66), ZenoStyle::dpiScaled(36));
    m_ui->btnAddOutput->setFixedSize(ZenoStyle::dpiScaled(66), ZenoStyle::dpiScaled(36));
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
    m_ui->outputsView->setAlternatingRowColors(true);
    m_ui->listConctrl->setFocusPolicy(Qt::NoFocus);
    //m_ui->paramsView->setFocusPolicy(Qt::NoFocus);
    resize(ZenoStyle::dpiScaled(900), ZenoStyle::dpiScaled(620));
    setFocusPolicy(Qt::ClickFocus);
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

void ZEditParamLayoutDlg::proxyModelSetData(const QModelIndex& index, const QVariant& newValue, int role)
{
    //TODO: ?
    //const QString& objPath = index.data(ROLE_OBJPATH).toString();
    m_paramsLayoutM_inputs->setData(index, newValue, role);
}

void ZEditParamLayoutDlg::onParamTreeDeleted()
{
    QModelIndex idx = m_ui->paramsView->currentIndex();
    bool bEditable = true;//TODO:
    if (!idx.isValid() || !idx.parent().isValid() || !bEditable)
        return;

    VPARAM_TYPE type = (VPARAM_TYPE)idx.data(ROLE_ELEMENT_TYPE).toInt();
    if (type == VPARAM_ROOT ||
        type == VPARAM_TAB && idx.data(Qt::DisplayRole).toString() == "Default" ||
        type == VPARAM_GROUP && idx.data(Qt::DisplayRole).toString() == "inputs")   //不允许删除默认root-tab-group输入
        return;
    m_paramsLayoutM_inputs->removeRow(idx.row(), idx.parent());
}

void ZEditParamLayoutDlg::onOutputsListDeleted()
{
    QModelIndex idx = m_ui->outputsView->currentIndex();
    bool bEditable = true;//TODO:
    if (!idx.isValid() || !bEditable)
        return;
    m_paramsLayoutM_outputs->removeRow(idx.row());
}

void ZEditParamLayoutDlg::onTreeCurrentChanged(const QModelIndex& current, const QModelIndex& previous)
{
    auto pCurrentItem = m_paramsLayoutM_inputs->itemFromIndex(current);
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
    }
    else if (type == VPARAM_GROUP)
    {
        m_ui->cbControl->setEnabled(false);
    }
    else if (type == VPARAM_PARAM)
    {
        QStandardItem* parentItem = pCurrentItem->parent();
        zeno::ParamControl ctrl = (zeno::ParamControl)pCurrentItem->data(ROLE_PARAM_CONTROL).toInt();
        const zeno::ParamType paramType = (zeno::ParamType)pCurrentItem->data(ROLE_PARAM_TYPE).toInt();
        const zeno::SocketType socketType = (zeno::SocketType)pCurrentItem->data(ROLE_SOCKET_TYPE).toInt();

        const QString& ctrlName = ctrl != zeno::NullControl ? getControl(ctrl, paramType).name : "";
        zeno::ControlProperty controlProperties = pCurrentItem->data(ROLE_PARAM_CTRL_PROPERTIES).value<zeno::ControlProperty>();

        const QString &parentName = parentItem->text();

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

        {
            BlockSignalScope scope(m_ui->cbControl);

            m_ui->cbControl->setEnabled(true);
            m_ui->cbControl->setCurrentText(ctrlName);
            //QString descType = UiHelper::getTypeDesc(paramType);
            if (ctrl == zeno::Seperator) 
            {
                m_ui->cbControl->setEnabled(false);
            }
        }

        {
            BlockSignalScope scope(m_ui->cbSocketType);
            if (parentName == "input") {
                m_ui->cbSocketType->setEnabled(true);
                if (zeno::NoSocket == socketType)
                    m_ui->cbSocketType->setCurrentText(tr("No Socket"));
                else if (zeno::PrimarySocket == socketType)
                    m_ui->cbSocketType->setCurrentText(tr("Primary Socket"));
                else if (zeno::ParamSocket == socketType)
                    m_ui->cbSocketType->setCurrentText(tr("Parameter Socket"));
            }
            else if (parentName == "output") {
                m_ui->cbSocketType->setEnabled(false);
            }
        }

        switchStackProperties(ctrl, pCurrentItem);
    }
}

void ZEditParamLayoutDlg::onOutputsListCurrentChanged(const QModelIndex& current, const QModelIndex& previous)
{
    auto pCurrentItem = m_paramsLayoutM_outputs->itemFromIndex(current);
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

    zeno::ParamControl ctrl = (zeno::ParamControl)pCurrentItem->data(ROLE_PARAM_CONTROL).toInt();
    const zeno::ParamType paramType = (zeno::ParamType)pCurrentItem->data(ROLE_PARAM_TYPE).toInt();
    const zeno::SocketType socketType = (zeno::SocketType)pCurrentItem->data(ROLE_SOCKET_TYPE).toInt();

    const QString& ctrlName = ctrl != zeno::NullControl ? getControl(ctrl, paramType).name : "";
    zeno::ControlProperty controlProperties = pCurrentItem->data(ROLE_PARAM_CTRL_PROPERTIES).value<zeno::ControlProperty>();

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

    QWidget* valueControl = zenoui::createWidget(deflVal, ctrl, paramType, cbSets, controlProperties);
    if (valueControl) {
        valueControl->setEnabled(bEditable);
        m_ui->gridLayout->addWidget(valueControl, rowValueControl, 1);
    }

    {
        BlockSignalScope scope(m_ui->cbControl);

        m_ui->cbControl->setEnabled(true);
        m_ui->cbControl->setCurrentText(ctrlName);
        //QString descType = UiHelper::getTypeDesc(paramType);
        if (ctrl == zeno::Seperator)
        {
            m_ui->cbControl->setEnabled(false);
        }
    }

    {
        BlockSignalScope scope(m_ui->cbSocketType);
        m_ui->cbSocketType->setEnabled(false);
    }

    switchStackProperties(ctrl, pCurrentItem);
}

void ZEditParamLayoutDlg::onBtnAddInputs()
{
    QModelIndex ctrlIdx = m_ui->listConctrl->currentIndex();
    if (!ctrlIdx.isValid())
        return;

    QModelIndex layerIdx = m_ui->paramsView->currentIndex();
    if (!layerIdx.isValid())
        return;

    QString ctrlName = ctrlIdx.data().toString();
    VPARAM_TYPE type = (VPARAM_TYPE)layerIdx.data(ROLE_ELEMENT_TYPE).toInt();
    auto pItem = m_paramsLayoutM_inputs->itemFromIndex(layerIdx);
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
        pNewItem->setData(VPARAM_TAB, ROLE_ELEMENT_TYPE);
        pNewItem->setData(newTabName, ROLE_PARAM_NAME);
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
        QString newGroupName = UiHelper::getUniqueName(existNames, "Group");
        auto pNewItem = new QStandardItem(newGroupName);
        pNewItem->setData(VPARAM_GROUP, ROLE_ELEMENT_TYPE);
        pNewItem->setData(newGroupName, ROLE_PARAM_NAME);
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
        if (layerIdx.data(Qt::DisplayRole).toString() == "inputs")  //判断是否和已有输出重名
            for (int r = 0; r < m_paramsLayoutM_outputs->rowCount(); r++)
                if (QStandardItem* pChildItem = m_paramsLayoutM_outputs->invisibleRootItem()->child(r))
                    if (newParamName == pChildItem->data(ROLE_PARAM_NAME).toString()) {
                        existNames.append(newParamName);
                        newParamName = UiHelper::getUniqueName(existNames, ctrl.name);
                    }
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
                std::array<float, 3> ranges = { 0.0, 100.0,1.0};
                zeno::ControlProperty pros;
                pros.ranges = ranges;
                pNewItem->setData(QVariant::fromValue(pros), ROLE_PARAM_CTRL_PROPERTIES);
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

void ZEditParamLayoutDlg::onBtnAddOutputs()
{
    QModelIndex ctrlIdx = m_ui->listConctrl->currentIndex();
    if (!ctrlIdx.isValid())
        return;
    QString ctrlName = ctrlIdx.data().toString();
    if (ctrlName == "Tab" || ctrlName == "Group")
        return;

    auto a = m_paramsLayoutM_outputs->rowCount();
    auto b = m_paramsLayoutM_outputs->columnCount();

    auto root = m_paramsLayoutM_outputs->invisibleRootItem();
    QStringList existNames;
    for (int r = 0; r < m_paramsLayoutM_outputs->rowCount(); r++)
    {
        QStandardItem* pChildItem = root->child(r);
        ZASSERT_EXIT(pChildItem);
        QString _name = pChildItem->data(ROLE_PARAM_NAME).toString();
        existNames.append(_name);
    }

    CONTROL_ITEM_INFO ctrl = getControlByName(ctrlName);
    QString newParamName = UiHelper::getUniqueName(existNames, ctrl.name);
    if (QStandardItem* pInputs = m_paramsLayoutM_inputs->item(0)->child(0)->child(0))   //和已有inputs参数名比较是否重名
        for (int r = 0; r < pInputs->rowCount(); r++)
            if (QStandardItem* pChildItem = pInputs->child(r))
                if (newParamName == pChildItem->data(ROLE_PARAM_NAME).toString()) {
                    existNames.append(newParamName);
                    newParamName = UiHelper::getUniqueName(existNames, ctrl.name);
                }
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
            std::array<float, 3> ranges = { 0.0, 100.0,1.0};
            zeno::ControlProperty pros;
            pros.ranges = ranges;
            pNewItem->setData(QVariant::fromValue(pros), ROLE_PARAM_CTRL_PROPERTIES);
            break;
        }
        case zeno::Seperator: 
        {
            //pNewItem->m_sockProp = SOCKPROP_GROUP_LINE;
            break;
        }
    }
    m_paramsLayoutM_outputs->appendRow(pNewItem);
    pNewItem->setData(getIcon(pNewItem), Qt::DecorationRole);
}

void ZEditParamLayoutDlg::switchStackProperties(int ctrl, QStandardItem* pItem)
{
    zeno::ControlProperty pros = pItem->data(ROLE_PARAM_CTRL_PROPERTIES).value<zeno::ControlProperty>();
    if (ctrl == zeno::Combobox) {
        if (pros.items.has_value()) {
                QStringList items;
                for (auto item : pros.items.value())
                    items.push_back(QString::fromStdString(item));
                QString value = pItem->data(ROLE_PARAM_VALUE).toString();
                for (int r = 0; r < items.size(); r++) {
                    QTableWidgetItem *newItem = new QTableWidgetItem(items[r]);
                    QLayoutItem *pLayoutItem = m_ui->gridLayout->itemAtPosition(rowValueControl, 1);
                    if (pLayoutItem) {
                        QComboBox *pControl = qobject_cast<QComboBox *>(pLayoutItem->widget());
                        if (pControl) {
                            pControl->setCurrentText(value);
                            proxyModelSetData(pItem->index(), value, ROLE_PARAM_VALUE);
                        }
                    }
                }
        }
    } 
    else if (ctrl == zeno::Slider ||
             ctrl == zeno::SpinBox ||
             ctrl == zeno::SpinBoxSlider ||
             ctrl == zeno::DoubleSpinBox)
    {
        if (!pros.ranges.has_value()) {
            std::array<float, 3> ranges = { 0.0, 100.0,1.0 };;
            pros.ranges = ranges;
            pItem->setData(QVariant::fromValue(pros), ROLE_PARAM_CTRL_PROPERTIES);
        }
    }
}

void ZEditParamLayoutDlg::onParamsViewParamDataChanged(const QModelIndex &topLeft, const QModelIndex &bottomRight, const QVector<int> &roles)
{
    if (roles.isEmpty())
        return;
    int role = roles[0];
    if (role == ROLE_PARAM_CONTROL) 
    {
        QStandardItem *item = m_paramsLayoutM_inputs->itemFromIndex(topLeft);
        QIcon icon = getIcon(item);
        item->setData(icon, Qt::DecorationRole);
    }
    if (role == ROLE_PARAM_NAME)
    {
        const QModelIndex& paramsViewCurrIdx = m_ui->paramsView->currentIndex();
        if (paramsViewCurrIdx.isValid() && paramsViewCurrIdx == topLeft) {

            QString newName = m_paramsLayoutM_inputs->data(topLeft, ROLE_PARAM_NAME).toString();
            disconnect(m_ui->editName, SIGNAL(editingFinished()), this, SLOT(onNameEditFinished()));
            m_ui->editName->setText(newName);
            connect(m_ui->editName, SIGNAL(editingFinished()), this, SLOT(onNameEditFinished()));
        }
    }
}

void ZEditParamLayoutDlg::onOutputsViewParamDataChanged(const QModelIndex& topLeft, const QModelIndex& bottomRight, const QVector<int>& roles)
{
    if (roles.isEmpty())
        return;
    int role = roles[0];
    if (role == ROLE_PARAM_NAME)
    {
        auto x = topLeft.data(ROLE_PARAM_NAME).toString();
        auto xx = m_paramsLayoutM_outputs->data(topLeft, ROLE_PARAM_NAME).toString();
        const QModelIndex& outputsViewCurrIdx = m_ui->outputsView->currentIndex();
        if (outputsViewCurrIdx.isValid() && outputsViewCurrIdx == topLeft) {
            QString newName = m_paramsLayoutM_outputs->data(topLeft, ROLE_PARAM_NAME).toString();
            disconnect(m_ui->editName, SIGNAL(editingFinished()), this, SLOT(onNameEditFinished()));
            m_ui->editName->setText(newName);
            connect(m_ui->editName, SIGNAL(editingFinished()), this, SLOT(onNameEditFinished()));
        }
    }
}

static QStandardItem* getItem(QStandardItem* pItem, const QString& uniqueName, int* targetIdx)
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
    const QModelIndex& paramsViewCurrIdx = m_ui->paramsView->currentIndex();
    const QModelIndex& outputsViewCurrIdx = m_ui->outputsView->currentIndex();
    QString newName = m_ui->editName->text();

    QStandardItem* currentItem;
    QString oldName;
    if (paramsViewCurrIdx.isValid()) {          //修改的参数来自paramsView
        currentItem = m_paramsLayoutM_inputs->itemFromIndex(paramsViewCurrIdx);
    }else if (outputsViewCurrIdx.isValid()) {   //修改的参数来自outputView
        currentItem = m_paramsLayoutM_outputs->itemFromIndex(outputsViewCurrIdx);
    }
    oldName = currentItem->data(ROLE_PARAM_NAME).toString();
    if (oldName != newName) {
        if (currentItem && m_isGlobalUniqueFunc(newName))
        {
            currentItem->setData(newName, ROLE_PARAM_NAME);
            currentItem->setText(newName);
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

    QStandardItem *pItem = m_paramsLayoutM_inputs->itemFromIndex(currIdx);

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

void ZEditParamLayoutDlg::onSocketTypeChanged(int idx)
{
    const QModelIndex& currIdx = m_ui->paramsView->currentIndex();
    if (!currIdx.isValid())
        return;

    QStandardItem* pItem = m_paramsLayoutM_inputs->itemFromIndex(currIdx);
    const QString& socketType = m_ui->cbSocketType->itemText(idx);
    if (socketType == tr("No Socket")) {
        pItem->setData(zeno::NoSocket, ROLE_SOCKET_TYPE);
    }
    else if (socketType == tr("Primary Socket")) {
        pItem->setData(zeno::PrimarySocket, ROLE_SOCKET_TYPE);
    }
    else if (socketType == tr("Parameter Socket")) {
        pItem->setData(zeno::ParamSocket, ROLE_SOCKET_TYPE);
    }
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
    auto pItem = m_paramsLayoutM_inputs->itemFromIndex(layerIdx);
    pItem->setData(type, ROLE_PARAM_TYPE);

    QVariant value = layerIdx.data(ROLE_PARAM_VALUE);
    if (!value.isValid())
        value = UiHelper::initDefaultValue(type);

    zeno::ControlProperty controlProperties = layerIdx.data(ROLE_PARAM_CTRL_PROPERTIES).value< zeno::ControlProperty>();
    cbSets.cbGetIndexData = [=]() -> QVariant { return UiHelper::initDefaultValue(type); };
    QWidget *valueControl = zenoui::createWidget(value, ctrl, type, cbSets, controlProperties);
    if (valueControl) {
        valueControl->setEnabled(true);
        m_ui->gridLayout->addWidget(valueControl, rowValueControl, 1);
        
        switchStackProperties(ctrl, pItem);
        pItem->setData(getIcon(pItem), Qt::DecorationRole);
    }
}

zeno::ParamsUpdateInfo ZEditParamLayoutDlg::getEdittedUpdateInfo() const
{
    return m_paramsUpdate;
}

bool ZEditParamLayoutDlg::eventFilter(QObject* obj, QEvent* event)
{
    if (obj == m_ui->paramsView && event->type() == QEvent::KeyPress)
    {
        QKeyEvent* e = static_cast<QKeyEvent*>(event);
        if (e->key() == Qt::Key_Delete) {
            onParamTreeDeleted();
            return true;
        }
    }
    if (obj == m_ui->outputsView && event->type() == QEvent::KeyPress)
    {
        QKeyEvent* e = static_cast<QKeyEvent*>(event);
        if (e->key() == Qt::Key_Delete)
        {
            onOutputsListDeleted();
            return true;
        }
    }
    return QDialog::eventFilter(obj, event);
}

void ZEditParamLayoutDlg::onApply()
{
    //temp case: for only inputs and outputs.
    m_paramsUpdate.clear();
    QStandardItem* pRoot = m_paramsLayoutM_inputs->item(0);
    QStandardItem* pDefautTab = pRoot->child(0);
    QStandardItem* pInputsGroup = pDefautTab->child(0);

    for (int i = 0; i < pInputsGroup->rowCount(); i++)
    {
        QStandardItem* pItem = pInputsGroup->child(i);
        zeno::ParamInfo param;

        param.bInput = true;
        param.control = (zeno::ParamControl)pItem->data(ROLE_PARAM_CONTROL).toInt();
        param.type = (zeno::ParamType)pItem->data(ROLE_PARAM_TYPE).toInt();
        param.defl = UiHelper::qvarToZVar(pItem->data(ROLE_PARAM_VALUE), param.type);
        param.name = pItem->data(ROLE_PARAM_NAME).toString().toStdString();
        param.tooltip = pItem->data(ROLE_PARAM_TOOLTIP).toString().toStdString();
        param.socketType = (zeno::SocketType)pItem->data(ROLE_SOCKET_TYPE).toInt();
        param.ctrlProps = pItem->data(ROLE_PARAM_CTRL_PROPERTIES).value<zeno::ControlProperty>();
        const QString& existName = pItem->data(ROLE_MAP_TO_PARAMNAME).toString();

        m_paramsUpdate.push_back({ param, existName.toStdString() });
    }
    for (int i = 0; i < m_paramsLayoutM_outputs->rowCount(); i++)
    {
        QStandardItem* pItem = m_paramsLayoutM_outputs->item(i);
        zeno::ParamInfo param;

        param.bInput = false;
        param.control = (zeno::ParamControl)pItem->data(ROLE_PARAM_CONTROL).toInt();
        param.type = (zeno::ParamType)pItem->data(ROLE_PARAM_TYPE).toInt();
        param.defl = UiHelper::qvarToZVar(pItem->data(ROLE_PARAM_VALUE), param.type);
        param.name = pItem->data(ROLE_PARAM_NAME).toString().toStdString();
        param.tooltip = pItem->data(ROLE_PARAM_TOOLTIP).toString().toStdString();
        param.socketType = (zeno::SocketType)pItem->data(ROLE_SOCKET_TYPE).toInt();
        param.ctrlProps = pItem->data(ROLE_PARAM_CTRL_PROPERTIES).value<zeno::ControlProperty>();
        const QString& existName = pItem->data(ROLE_MAP_TO_PARAMNAME).toString();

        m_paramsUpdate.push_back({ param, existName.toStdString() });
    }

    //m_model->batchModifyParams(params);
}

void ZEditParamLayoutDlg::onOk()
{
    accept();
    if (m_paramsUpdate.empty())
        onApply();
}

void ZEditParamLayoutDlg::onCancel()
{
    reject();
}