#include "zeditparamlayoutdlg.h"
#include "ui_zeditparamlayoutdlg.h"
#include "zassert.h"
#include <zenomodel/include/uihelper.h>
#include "zmapcoreparamdlg.h"
#include <zenomodel/include/uihelper.h>
#include "zenoapplication.h"
#include <zenomodel/include/graphsmanagment.h>
#include <zenomodel/include/nodeparammodel.h>
#include <zenomodel/include/panelparammodel.h>
#include <zenomodel/include/nodesmgr.h>
#include <zenoui/comctrl/zwidgetfactory.h>
#include <zenomodel/include/globalcontrolmgr.h>
#include "variantptr.h"
#include <zenomodel/include/command.h>
#include "iotags.h"
#include <zenoui/style/zenostyle.h>


static CONTROL_ITEM_INFO controlList[] = {
    {"Tab", CONTROL_NONE, "", ":/icons/parameter_control_tab.svg"},
    {"Group", CONTROL_NONE, "", ":/icons/parameter_control_group.svg"},
    {"Integer",             CONTROL_INT,            "int", ":/icons/parameter_control_integer.svg"},
    {"Float",               CONTROL_FLOAT,          "float", ":/icons/parameter_control_float.svg"},
    {"String",              CONTROL_STRING,         "string", ":/icons/parameter_control_string.svg"},
    {"Boolean",             CONTROL_BOOL,           "bool", ":/icons/parameter_control_boolean.svg"},
    {"Multiline String",    CONTROL_MULTILINE_STRING, "string", ":/icons/parameter_control_string.svg"},
    {"read path",           CONTROL_READPATH,       "string", ":/icons/parameter_control_fold.svg"},
    {"write path",          CONTROL_WRITEPATH,      "string", ":/icons/parameter_control_fold.svg"},
    {"Enum",                CONTROL_ENUM,           "string", ":/icons/parameter_control_enum.svg"},
    {"Float Vector 4",      CONTROL_VEC4_FLOAT,     "vec4f", ":/icons/parameter_control_floatVector4.svg"},
    {"Float Vector 3",      CONTROL_VEC3_FLOAT,     "vec3f", ":/icons/parameter_control_floatVector3.svg"},
    {"Float Vector 2",      CONTROL_VEC2_FLOAT,     "vec2f", ":/icons/parameter_control_floatVector2.svg"},
    {"Integer Vector 4",    CONTROL_VEC4_INT,       "vec4i", ":/icons/parameter_control_integerVector4.svg"},
    {"Integer Vector 3",    CONTROL_VEC3_INT,       "vec3i", ":/icons/parameter_control_integerVector3.svg"},
    {"Integer Vector 2",    CONTROL_VEC2_INT,       "vec2i", ":/icons/parameter_control_integerVector2.svg"},
    {"Color",               CONTROL_COLOR,          "color", ":/icons/parameter_control_color.svg"},
    {"Curve",               CONTROL_CURVE,          "curve", ":/icons/parameter_control_curve.svg"},
    {"SpinBox",             CONTROL_HSPINBOX,       "int", ":/icons/parameter_control_spinbox.svg"},
    {"DoubleSpinBox", CONTROL_HDOUBLESPINBOX, "float", ":/icons/parameter_control_spinbox.svg"},
    {"Slider",              CONTROL_HSLIDER,        "int", ":/icons/parameter_control_slider.svg"},
    {"SpinBoxSlider",       CONTROL_SPINBOX_SLIDER, "int", ":/icons/parameter_control_slider.svg"},
    {"Divider", CONTROL_GROUP_LINE, "", ":/icons/parameter_control_divider.svg"},
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


ParamTreeItemDelegate::ParamTreeItemDelegate(ViewParamModel *model, QObject *parent)
    : QStyledItemDelegate(parent),
    m_model(model) 
{
}

ParamTreeItemDelegate::~ParamTreeItemDelegate()
{
}

QWidget* ParamTreeItemDelegate::createEditor(QWidget* parent, const QStyleOptionViewItem& option, const QModelIndex& index) const
{
    bool bEditable = m_model->isEditable(index);
    if (!bEditable)
        return nullptr;
    return QStyledItemDelegate::createEditor(parent, option, index);
}

void ParamTreeItemDelegate::setModelData(QWidget *editor, QAbstractItemModel *model, const QModelIndex &index) const 
{
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
}

void ParamTreeItemDelegate::paint(QPainter *painter, const QStyleOptionViewItem &option,
                                  const QModelIndex &index) const {
    QStyledItemDelegate::paint(painter, option, index);
}


ZEditParamLayoutDlg::ZEditParamLayoutDlg(QStandardItemModel* pModel, bool bNodeUI, const QPersistentModelIndex& nodeIdx, IGraphsModel* pGraphsModel, QWidget* parent)
    : QDialog(parent)
    , m_model(nullptr)
    , m_proxyModel(nullptr)
    , m_nodeIdx(nodeIdx)
    , m_pGraphsModel(pGraphsModel)
    , m_bSubgraphNode(false)
    , m_bNodeUI(bNodeUI)
{
    m_ui = new Ui::EditParamLayoutDlg;
    m_ui->setupUi(this);
    initUI();

    for (int i = 0; i < sizeof(controlList) / sizeof(CONTROL_ITEM_INFO); i++)
    {
        if (bNodeUI && (controlList[i].ctrl == CONTROL_HSLIDER || controlList[i].ctrl == CONTROL_SPINBOX_SLIDER))
            continue;
        else if (!bNodeUI && controlList[i].ctrl == CONTROL_GROUP_LINE)
            continue;
        QListWidgetItem *item = new QListWidgetItem(controlList[i].name, m_ui->listConctrl);
        item->setIcon(QIcon(controlList[i].icon));
        m_ui->listConctrl->addItem(item);
        if (controlList[i].ctrl == CONTROL_NONE)
            continue;
        m_ui->cbControl->addItem(controlList[i].name);
    }

    m_ui->cbTypes->addItems(UiHelper::getCoreTypeList());

    m_model = qobject_cast<ViewParamModel*>(pModel);
    ZASSERT_EXIT(m_model);

    m_bSubgraphNode = m_pGraphsModel->IsSubGraphNode(m_nodeIdx) && m_model->isNodeModel();
    m_subgIdx = m_nodeIdx.data(ROLE_SUBGRAPH_IDX).toModelIndex();

    if (bNodeUI)
    {
        m_ui->m_coreMappingWidget->hide();
        m_proxyModel = new NodeParamModel(m_subgIdx, m_model->nodeIdx(), m_pGraphsModel, true, this);
    }
    else
    {
        m_proxyModel = new PanelParamModel(m_model->nodeIdx(), m_pGraphsModel, this);
    }

    m_proxyModel->clone(m_model);
    initDescValueForProxy();
    initIcon(m_proxyModel->invisibleRootItem());

    m_ui->paramsView->setModel(m_proxyModel);
    m_ui->paramsView->setItemDelegate(new ParamTreeItemDelegate(m_proxyModel, m_ui->paramsView));

    QItemSelectionModel* selModel = m_ui->paramsView->selectionModel();
    connect(selModel, SIGNAL(currentChanged(const QModelIndex &, const QModelIndex &)), this,
            SLOT(onTreeCurrentChanged(const QModelIndex &, const QModelIndex &)));
    QModelIndex selIdx = selModel->currentIndex();
    const QModelIndex& wtfIdx = m_proxyModel->index(0, 0);
    selModel->setCurrentIndex(wtfIdx, QItemSelectionModel::SelectCurrent);
    m_ui->paramsView->expandAll();

    connect(m_ui->editName, SIGNAL(editingFinished()), this, SLOT(onNameEditFinished()));
    connect(m_ui->editLabel, SIGNAL(editingFinished()), this, SLOT(onLabelEditFinished()));
    connect(m_ui->btnAdd, SIGNAL(clicked()), this, SLOT(onBtnAdd()));
    connect(m_ui->btnApply, SIGNAL(clicked()), this, SLOT(onApply()));
    connect(m_ui->btnOk, SIGNAL(clicked()), this, SLOT(onOk()));
    connect(m_ui->btnCancel, SIGNAL(clicked()), this, SLOT(onCancel()));

    connect(m_proxyModel, SIGNAL(editNameChanged(const QModelIndex&, const QString&, const QString&)),
            this, SLOT(onProxyItemNameChanged(const QModelIndex&, const QString&, const QString&)));

    QShortcut* shortcut = new QShortcut(QKeySequence(Qt::Key_Delete), m_ui->paramsView);
    connect(shortcut, SIGNAL(activated()), this, SLOT(onParamTreeDeleted()));

    connect(m_ui->btnChooseCoreParam, SIGNAL(clicked(bool)), this, SLOT(onChooseParamClicked()));
    connect(m_ui->editMin, SIGNAL(editingFinished()), this, SLOT(onMinEditFinished()));
    connect(m_ui->editMax, SIGNAL(editingFinished()), this, SLOT(onMaxEditFinished()));
    connect(m_ui->editStep, SIGNAL(editingFinished()), this, SLOT(onStepEditFinished()));
    connect(m_ui->cbControl, SIGNAL(currentIndexChanged(int)), this, SLOT(onControlItemChanged(int)));
    connect(m_ui->cbTypes, SIGNAL(currentIndexChanged(int)), this, SLOT(onTypeItemChanged(int)));

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
}

void ZEditParamLayoutDlg::initUI() 
{
    m_ui->labelCreateControl->setProperty("cssClass", "bold");
    m_ui->labelPrameter->setProperty("cssClass", "bold");
    m_ui->labelSetting->setProperty("cssClass", "bold");
    m_ui->label_11->setProperty("cssClass", "bold");
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
    int control = pItem->data(ROLE_PARAM_CTRL).toInt();
    int type = pItem->data(ROLE_VPARAM_TYPE).toInt();
    if (type == VPARAM_TAB) 
    {
        return QIcon(":/icons/parameter_control_tab.svg");
    } 
    else if (type == VPARAM_GROUP) 
    {
        return QIcon(":/icons/parameter_control_group.svg");
    } 
    else if (type != VPARAM_ROOT) 
    {
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

void ZEditParamLayoutDlg::initDescValueForProxy() {
    if (m_bNodeUI && m_bSubgraphNode)
    {
        NODE_DESC desc;
        const QString& subnetNodeName = m_nodeIdx.data(ROLE_OBJNAME).toString();
        m_pGraphsModel->getDescriptor(subnetNodeName, desc);
        NodeParamModel* nodeParams = qobject_cast<NodeParamModel*>(m_proxyModel);
        ZASSERT_EXIT(nodeParams);
        VParamItem* inputs = nodeParams->getInputs();

        for (INPUT_SOCKET inSocket : desc.inputs)
        {
            VParamItem* pItem = inputs->getItem(inSocket.info.name);
            ZASSERT_EXIT(pItem);
            pItem->setData(inSocket.info.defaultValue, ROLE_PARAM_VALUE);
        }
    }
}

void ZEditParamLayoutDlg::onComboTableItemsCellChanged(int row, int column)
{
    //dump to item.
    QModelIndex layerIdx = m_ui->paramsView->currentIndex();
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

    CONTROL_PROPERTIES properties = layerIdx.data(ROLE_VPARAM_CTRL_PROPERTIES).value<CONTROL_PROPERTIES>();
    properties["items"] = lst;

    proxyModelSetData(layerIdx, properties, ROLE_VPARAM_CTRL_PROPERTIES);
    proxyModelSetData(layerIdx, lst[0], ROLE_PARAM_VALUE);
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
        }
    }
}

void ZEditParamLayoutDlg::proxyModelSetData(const QModelIndex& index, const QVariant& newValue, int role)
{
    const QString& objPath = index.data(ROLE_OBJPATH).toString();
    m_proxyModel->setData(index, newValue, role);
}

void ZEditParamLayoutDlg::onParamTreeDeleted()
{
    if (m_ui->itemsTable->hasFocus()) {
        int row = m_ui->itemsTable->currentRow();
        if (row < m_ui->itemsTable->rowCount() - 1)
            m_ui->itemsTable->removeRow(row);
    } else {
        QModelIndex idx = m_ui->paramsView->currentIndex();
        bool bEditable = m_proxyModel->isEditable(idx);
        if (!idx.isValid() || !idx.parent().isValid() || !bEditable)
            return;
        m_proxyModel->removeRow(idx.row(), idx.parent());
    }
}

void ZEditParamLayoutDlg::onTreeCurrentChanged(const QModelIndex& current, const QModelIndex& previous)
{
    VParamItem* pCurrentItem = static_cast<VParamItem*>(m_proxyModel->itemFromIndex(current));
    if (!pCurrentItem)  return;

    const QString& name = pCurrentItem->data(ROLE_VPARAM_NAME).toString();
    m_ui->editName->setText(name);
    bool bEditable = m_proxyModel->isEditable(current);
    m_ui->editName->setEnabled(bEditable);

    m_ui->editLabel->setText(pCurrentItem->data(ROLE_VPARAM_TOOLTIP).toString());

    //delete old control.
    QLayoutItem* pLayoutItem = m_ui->gridLayout->itemAtPosition(rowValueControl, 1);
    if (pLayoutItem)
    {
        QWidget* pControlWidget = pLayoutItem->widget();
        delete pControlWidget;
    }

    VPARAM_TYPE type = (VPARAM_TYPE)pCurrentItem->data(ROLE_VPARAM_TYPE).toInt();
    if (type == VPARAM_TAB)
    {
        m_ui->cbControl->setEnabled(false);
        m_ui->cbTypes->setEnabled(false);
        m_ui->stackProperties->setCurrentIndex(0);
    }
    else if (type == VPARAM_GROUP || pCurrentItem->m_ctrl == CONTROL_GROUP_LINE)
    {
        m_ui->cbControl->setEnabled(false);
        m_ui->cbTypes->setEnabled(false);
        m_ui->stackProperties->setCurrentIndex(0);
        m_ui->cbControl->setCurrentText("");
        m_ui->cbTypes->setCurrentText("");
    }
    else if (type == VPARAM_PARAM)
    {
        QStandardItem* parentItem = pCurrentItem->parent();
        PARAM_CONTROL ctrl = pCurrentItem->m_ctrl;
        const QString &ctrlName = getControl(ctrl).name;
        QVariant controlProperties = pCurrentItem->data(ROLE_VPARAM_CTRL_PROPERTIES);
        const QString &dataType = pCurrentItem->data(ROLE_PARAM_TYPE).toString();
        bool bCoreParam = pCurrentItem->data(ROLE_VPARAM_IS_COREPARAM).toBool();
        if (m_bSubgraphNode || bCoreParam) 
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
                deflVal = UiHelper::initVariantByControl(ctrl);

            cbSets.cbGetIndexData = [=]() -> QVariant {
                if (!pCurrentItem->data(ROLE_PARAM_VALUE).isValid()) {
                    return UiHelper::initVariantByControl(ctrl);
                }
                return pCurrentItem->data(ROLE_PARAM_VALUE);
            };
            QWidget *valueControl = zenoui::createWidget(deflVal, ctrl, dataType, cbSets, controlProperties);
            if (valueControl) {
                valueControl->setEnabled(bEditable);
                m_ui->gridLayout->addWidget(valueControl, rowValueControl, 1);
            }
        }

        {
            BlockSignalScope scope(m_ui->cbControl);
            BlockSignalScope scope2(m_ui->cbTypes);

            m_ui->cbControl->setEnabled(bEditable);
            m_ui->cbControl->clear();
            QStringList items = UiHelper::getControlLists(dataType, m_bNodeUI);
            m_ui->cbControl->addItems(items);
            m_ui->cbControl->setCurrentText(ctrlName);

            m_ui->cbTypes->setCurrentText(dataType);
            m_ui->cbTypes->setEnabled(bEditable);
        }
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
            m_ui->editCoreParamType->setText(dataType);
        }
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
    VPARAM_TYPE type = (VPARAM_TYPE)layerIdx.data(ROLE_VPARAM_TYPE).toInt();
    VParamItem* pItem = static_cast<VParamItem*>(m_proxyModel->itemFromIndex(layerIdx));
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
            QMessageBox::information(this, tr("Error"), tr("create tab needs to place under the root"));
            return;
        }
        QString newTabName = UiHelper::getUniqueName(existNames, "Tab");
        VParamItem* pNewItem = new VParamItem(VPARAM_TAB, newTabName);
        pItem->appendRow(pNewItem);

        QString objPath = pItem->data(ROLE_OBJPATH).toString();
        VPARAM_INFO vParam = pNewItem->exportParamInfo();
        pNewItem->setData(getIcon(pNewItem), Qt::DecorationRole);
    }
    else if (ctrlName == "Group")
    {
        if (type != VPARAM_TAB)
        {
            QMessageBox::information(this, tr("Error "), tr("create group needs to place under the tab"));
            return;
        }
        QString newGroup = UiHelper::getUniqueName(existNames, "Group");
        VParamItem* pNewItem = new VParamItem(VPARAM_GROUP, newGroup);
        pItem->appendRow(pNewItem);

        QString objPath = pItem->data(ROLE_OBJPATH).toString();
        VPARAM_INFO vParam = pNewItem->exportParamInfo();
        pNewItem->setData(getIcon(pNewItem), Qt::DecorationRole);
    }
    else
    {
        if (type != VPARAM_GROUP)
        {
            QMessageBox::information(this, tr("Error "), tr("create control needs to place under the group"));
            return;
        }
        bool bEditable = m_proxyModel->isEditable(layerIdx) || m_bSubgraphNode;
        if (!bEditable) {
            QMessageBox::information(this, tr("Error "), tr("The Group cannot be edited"));
            return;
        }
        CONTROL_ITEM_INFO ctrl = getControlByName(ctrlName);
        QString newItem = UiHelper::getUniqueName(existNames, ctrl.name);
        VParamItem* pNewItem = new VParamItem(VPARAM_PARAM, newItem);
        pNewItem->setData(ctrl.ctrl, ROLE_PARAM_CTRL);
        pNewItem->setData(ctrl.defaultType, ROLE_PARAM_TYPE);
        pNewItem->setData(UiHelper::initVariantByControl(ctrl.ctrl), ROLE_PARAM_VALUE);

        //init properties.
        switch (ctrl.ctrl)
        {
            case CONTROL_SPINBOX_SLIDER:
            case CONTROL_HSPINBOX:
            case CONTROL_HDOUBLESPINBOX:
            case CONTROL_HSLIDER:
            {
                CONTROL_PROPERTIES properties;
                properties["step"] = 1;
                properties["min"] = 0;
                properties["max"] = 100;
                pNewItem->setData(properties, ROLE_VPARAM_CTRL_PROPERTIES);
                break;
            }
            case CONTROL_GROUP_LINE: 
            {
                pNewItem->m_sockProp = SOCKPROP_GROUP_LINE;
                break;
            }
        }
        pItem->appendRow(pNewItem);
        pNewItem->setData(getIcon(pNewItem), Qt::DecorationRole);
    }
}

void ZEditParamLayoutDlg::recordSubInputCommands(bool bSubInput, VParamItem* pItem)
{
}

void ZEditParamLayoutDlg::switchStackProperties(int ctrl, VParamItem* pItem) 
{
    QVariant controlProperties = pItem->data(ROLE_VPARAM_CTRL_PROPERTIES);
    if (ctrl == CONTROL_ENUM) {
        CONTROL_PROPERTIES pros = controlProperties.toMap();
        
        if (pros.find("items") != pros.end()) {
                const QStringList &items = pros["items"].toStringList();
                m_ui->itemsTable->setRowCount(items.size() + 1);
                for (int r = 0; r < items.size(); r++) {
                QTableWidgetItem *newItem = new QTableWidgetItem(items[r]);
                m_ui->itemsTable->setItem(r, 0, newItem);
                }
        } else {
                m_ui->itemsTable->setRowCount(1);
        }
        m_ui->stackProperties->setCurrentIndex(1);
    } else if (ctrl == CONTROL_HSLIDER || ctrl == CONTROL_HSPINBOX 
                || ctrl == CONTROL_SPINBOX_SLIDER || ctrl == CONTROL_HDOUBLESPINBOX) {
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
                pItem->setData(properties, ROLE_VPARAM_CTRL_PROPERTIES);
        }
        m_ui->editStep->setText(QString::number(info.step));
        m_ui->editMin->setText(QString::number(info.min));
        m_ui->editMax->setText(QString::number(info.max));
    } else {
        m_ui->stackProperties->setCurrentIndex(0);
    }
}

void ZEditParamLayoutDlg::addControlGroup(bool bInput, const QString &name, PARAM_CONTROL ctrl) 
{
    NODE_DESC desc;
    QString subnetNodeName = m_nodeIdx.data(ROLE_OBJNAME).toString();
    m_pGraphsModel->getDescriptor(subnetNodeName, desc);
    SOCKET_INFO info;
    info.control = ctrl;
    info.name = name;
    info.type = "group-line";
    info.sockProp = SOCKPROP_GROUP_LINE;
    if (bInput) {
        desc.inputs[name].info = info;
    } else {
        desc.outputs[name].info = info;
    }
    m_pGraphsModel->updateSubgDesc(subnetNodeName, desc);
    //sync to all subgraph nodes.
    QModelIndexList subgNodes = m_pGraphsModel->findSubgraphNode(subnetNodeName);
    for (QModelIndex subgNode : subgNodes) 
    {
        if (subgNode == m_nodeIdx)
                continue;
        NodeParamModel *nodeParams = QVariantPtr<NodeParamModel>::asPtr(subgNode.data(ROLE_NODE_PARAMS));
        nodeParams->setAddParam(bInput ? PARAM_INPUT : PARAM_OUTPUT, name, "", QVariant(), ctrl, QVariant(),
                                SOCKPROP_NORMAL);
    }
}

void ZEditParamLayoutDlg::delControlGroup(bool bInput, const QString &name) 
{
    NODE_DESC desc;
    QString subnetNodeName = m_nodeIdx.data(ROLE_OBJNAME).toString();
    m_pGraphsModel->getDescriptor(subnetNodeName, desc);
    if (bInput) {
        ZASSERT_EXIT(desc.inputs.find(name) != desc.inputs.end());
        desc.inputs.remove(name);
    } else {
        ZASSERT_EXIT(desc.outputs.find(name) != desc.outputs.end());
        desc.outputs.remove(name);
    }
    m_pGraphsModel->updateSubgDesc(subnetNodeName, desc);

    QModelIndexList subgNodes = m_pGraphsModel->findSubgraphNode(subnetNodeName);
    for (QModelIndex subgNode : subgNodes) 
    {
        if (subgNode == m_nodeIdx)
            continue;
        NodeParamModel *nodeParams = QVariantPtr<NodeParamModel>::asPtr(subgNode.data(ROLE_NODE_PARAMS));
        nodeParams->removeParam(bInput ? PARAM_INPUT : PARAM_OUTPUT, name);
    }
}

void ZEditParamLayoutDlg::updateControlGroup(bool bInput, const QString &newName, const QString &oldName, PARAM_CONTROL ctrl, int row) 
{
    NODE_DESC desc;
    QString subnetNodeName = m_nodeIdx.data(ROLE_OBJNAME).toString();
    m_pGraphsModel->getDescriptor(subnetNodeName, desc);
    SOCKET_INFO info;
    info.control = ctrl;
    info.name = newName;
    info.type = "group-line";
    if (bInput) {
        desc.inputs[newName].info = info;
        ZASSERT_EXIT(desc.inputs.find(oldName) != desc.inputs.end());
        desc.inputs.remove(oldName);
        if (desc.inputs.size() - 1!= row)
            desc.inputs.move(desc.inputs.size() - 1, row);
    } else {
        desc.outputs[newName].info = info;
        ZASSERT_EXIT(desc.outputs.find(oldName) != desc.outputs.end());
        desc.outputs.remove(oldName);
        if (desc.outputs.size() - 1 != row)
            desc.outputs.move(desc.outputs.size() - 1, row);
    }
    m_pGraphsModel->updateSubgDesc(subnetNodeName, desc);

    QModelIndexList subgNodes = m_pGraphsModel->findSubgraphNode(subnetNodeName);
    for (QModelIndex subgNode : subgNodes) 
    {
        NodeParamModel *nodeParams = QVariantPtr<NodeParamModel>::asPtr(subgNode.data(ROLE_NODE_PARAMS));
        QModelIndex index = nodeParams->getParam(bInput ? PARAM_INPUT : PARAM_OUTPUT, oldName);
        nodeParams->setData(index, newName, ROLE_PARAM_NAME);
    }
}



void ZEditParamLayoutDlg::onProxyItemNameChanged(const QModelIndex& itemIdx, const QString& oldPath, const QString& newName)
{
    //ViewParamSetDataCommand *pCmd = new ViewParamSetDataCommand(m_pGraphsModel, oldPath, newName, ROLE_VPARAM_NAME);
    //m_commandSeq.append(pCmd);
    if (m_ui->editName->text() != newName)
        m_ui->editName->setText(newName);
}

void ZEditParamLayoutDlg::onNameEditFinished()
{
    const QModelIndex& currIdx = m_ui->paramsView->currentIndex();
    if (!currIdx.isValid())
        return;

    QStandardItem* pItem = m_proxyModel->itemFromIndex(currIdx);
    QString oldPath = pItem->data(ROLE_OBJPATH).toString();
    QString newName = m_ui->editName->text();
    QString oldName = pItem->data(ROLE_VPARAM_NAME).toString();
    if (oldName != newName) {
        int dstRow = 0;
        VParamItem *pTargetGroup = static_cast<VParamItem *>(pItem->parent());
        if (pTargetGroup && !pTargetGroup->getItem(newName, &dstRow))
        {
            pItem->setData(newName, ROLE_VPARAM_NAME);
            onProxyItemNameChanged(pItem->index(), oldPath, newName);
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

    QStandardItem *pItem = m_proxyModel->itemFromIndex(currIdx);

    ZASSERT_EXIT(pItem);
    QString oldText = pItem->data(ROLE_VPARAM_NAME).toString();
    QString newText = m_ui->editLabel->text();
    if (oldText == newText)
        return;
    pItem->setData(newText, ROLE_VPARAM_TOOLTIP);
}

void ZEditParamLayoutDlg::onHintEditFinished()
{

}

void ZEditParamLayoutDlg::onMinEditFinished()
{
    QModelIndex layerIdx = m_ui->paramsView->currentIndex();
    if (!layerIdx.isValid() && layerIdx.data(ROLE_VPARAM_TYPE) != VPARAM_PARAM)
        return;

    CONTROL_PROPERTIES properties = layerIdx.data(ROLE_VPARAM_CTRL_PROPERTIES).value<CONTROL_PROPERTIES>();
    qreal from = m_ui->editMin->text().toDouble();
    properties["min"] = from;
    proxyModelSetData(layerIdx, properties, ROLE_VPARAM_CTRL_PROPERTIES);
}

void ZEditParamLayoutDlg::onMaxEditFinished()
{
    QModelIndex layerIdx = m_ui->paramsView->currentIndex();
    if (!layerIdx.isValid() && layerIdx.data(ROLE_VPARAM_TYPE) != VPARAM_PARAM)
        return;

    CONTROL_PROPERTIES properties = layerIdx.data(ROLE_VPARAM_CTRL_PROPERTIES).value<CONTROL_PROPERTIES>();
    qreal to = m_ui->editMax->text().toDouble();
    properties["max"] = to;
    proxyModelSetData(layerIdx, properties, ROLE_VPARAM_CTRL_PROPERTIES);
}

void ZEditParamLayoutDlg::onControlItemChanged(int idx)
{
    const QString& controlName = m_ui->cbControl->itemText(idx);
    PARAM_CONTROL ctrl = UiHelper::getControlByDesc(controlName);
    QModelIndex layerIdx = m_ui->paramsView->currentIndex();
    if (!layerIdx.isValid() && layerIdx.data(ROLE_VPARAM_TYPE) != VPARAM_PARAM)
        return;

    proxyModelSetData(layerIdx, ctrl, ROLE_PARAM_CTRL);

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
    const QString &dataType = m_ui->cbTypes->itemText(idx);
    QVariant value = UiHelper::initVariantByControl(ctrl);
    QVariant controlProperties = layerIdx.data(ROLE_VPARAM_CTRL_PROPERTIES);
    cbSets.cbGetIndexData = [=]() -> QVariant { return UiHelper::initVariantByControl(ctrl); };
    QWidget *valueControl = zenoui::createWidget(value, ctrl, dataType, cbSets, controlProperties);
    if (valueControl) {
        valueControl->setEnabled(m_pGraphsModel->IsSubGraphNode(m_nodeIdx));
        m_ui->gridLayout->addWidget(valueControl, rowValueControl, 1);
        VParamItem *pItem = static_cast<VParamItem *>(m_proxyModel->itemFromIndex(layerIdx));
        switchStackProperties(ctrl, pItem);
        pItem->setData(getIcon(pItem), Qt::DecorationRole);
    }
}

void ZEditParamLayoutDlg::onTypeItemChanged(int idx)
{
    const QString& dataType = m_ui->cbTypes->itemText(idx);
    QModelIndex layerIdx = m_ui->paramsView->currentIndex();
    if (!layerIdx.isValid() && layerIdx.data(ROLE_VPARAM_TYPE) != VPARAM_PARAM)
        return;

    VParamItem* pItem = static_cast<VParamItem*>(m_proxyModel->itemFromIndex(layerIdx));
    pItem->setData(dataType, ROLE_PARAM_TYPE);
    pItem->setData(UiHelper::getControlByType(dataType), ROLE_PARAM_CTRL);
    pItem->setData(UiHelper::initVariantByControl(pItem->m_ctrl), ROLE_PARAM_VALUE);

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
    QVariant controlProperties = layerIdx.data(ROLE_VPARAM_CTRL_PROPERTIES);
    cbSets.cbGetIndexData = [=]() -> QVariant { return pItem->data(ROLE_PARAM_VALUE); };
    QWidget *valueControl = zenoui::createWidget(pItem->data(ROLE_PARAM_VALUE), pItem->m_ctrl, dataType, cbSets, controlProperties);
    if (valueControl) {
        valueControl->setEnabled(m_pGraphsModel->IsSubGraphNode(m_nodeIdx));
        m_ui->gridLayout->addWidget(valueControl, rowValueControl, 1);
        switchStackProperties(pItem->m_ctrl, pItem);
        pItem->setData(getIcon(pItem), Qt::DecorationRole);
    }
    //update control list
    QStringList items = UiHelper::getControlLists(dataType, m_bNodeUI);
    m_ui->cbControl->clear();
    m_ui->cbControl->addItems(items);
}

void ZEditParamLayoutDlg::onStepEditFinished()
{
    QModelIndex layerIdx = m_ui->paramsView->currentIndex();
    if (!layerIdx.isValid() && layerIdx.data(ROLE_VPARAM_TYPE) != VPARAM_PARAM)
        return;

    CONTROL_PROPERTIES properties = layerIdx.data(ROLE_VPARAM_CTRL_PROPERTIES).value<CONTROL_PROPERTIES>();
    qreal step = m_ui->editStep->text().toDouble();
    properties["step"] = step;

    const QString &objPath = layerIdx.data(ROLE_OBJPATH).toString();
    //ViewParamSetDataCommand *pCommand = new ViewParamSetDataCommand(m_pGraphsModel, objPath, properties, ROLE_VPARAM_CTRL_PROPERTIES);
    //m_commandSeq.append(pCommand);

    m_proxyModel->setData(layerIdx, properties, ROLE_VPARAM_CTRL_PROPERTIES);
}

void ZEditParamLayoutDlg::onChooseParamClicked()
{
    ZMapCoreparamDlg dlg(m_nodeIdx);
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

            PARAM_CONTROL newCtrl = UiHelper::getControlByType(typeDesc);
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

void ZEditParamLayoutDlg::applyForItem(QStandardItem* proxyItem, QStandardItem* appliedItem)
{
    //oldItem: proxy.

    QString subgName;
    QModelIndex subgIdx;
    bool bSubInput = false;
    bool bApplySubnetParam = m_bSubgraphNode && proxyItem->data(ROLE_VPARAM_TYPE) == VPARAM_GROUP;
    if (bApplySubnetParam)
    {
        VParamItem *pGroup = static_cast<VParamItem *>(proxyItem);
        bSubInput = pGroup->m_name == iotags::params::node_inputs;
        subgName = m_nodeIdx.data(ROLE_OBJNAME).toString();
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
        const PARAM_CONTROL ctrl = (PARAM_CONTROL)pCurrent->data(ROLE_PARAM_CTRL).toInt();
        QVariant ctrlProperties;
        if (pCurrent->m_customData.find(ROLE_VPARAM_CTRL_PROPERTIES) != pCurrent->m_customData.end()) 
		{
            ctrlProperties = pCurrent->m_customData[ROLE_VPARAM_CTRL_PROPERTIES];
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
                    VParamItem* pGroup = bSubInput ? nodeParams->getInputs() : nodeParams->getOutputs();
                    ZASSERT_EXIT(pGroup);
                    QModelIndex parent = pGroup->index();
                    nodeParams->moveRow(parent, srcRow, parent, dstRow);
                }

                //reacquire pTarget, because the implementation of moveRow is simplily
                pTarget = static_cast<VParamItem*>(appliedItem->child(r));
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
                        const QModelIndex &deflIdx = nodeParams->getParam(PARAM_PARAM, "defl");
                        //update the control on "defl" in SubInput/SubOutput.
                        m_pGraphsModel->ModelSetData(deflIdx, ctrl, ROLE_PARAM_CTRL);
                    } 
                    pTarget->setData(ctrl, ROLE_PARAM_CTRL);
                }

	            //control properties
                bool isChanged = pTarget->m_customData[ROLE_VPARAM_CTRL_PROPERTIES] != ctrlProperties;
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
                            m_pGraphsModel->ModelSetData(deflIdx, ctrlProperties, ROLE_VPARAM_CTRL_PROPERTIES);
                        }
                    }
                    pTarget->setData(ctrlProperties, ROLE_VPARAM_CTRL_PROPERTIES);
                }
                //core mapping
                if (pCurrent->m_index != pTarget->m_index) 
                {
                    pTarget->mapCoreParam(pCurrent->m_index);
                }
                //tool tip
                QVariant newTip;
                if (pCurrent->m_customData.find(ROLE_VPARAM_TOOLTIP) != pCurrent->m_customData.end()) {
                    newTip = pCurrent->m_customData[ROLE_VPARAM_TOOLTIP];
                }
                QVariant oldTip = pTarget->m_customData[ROLE_VPARAM_TOOLTIP];
                if (oldTip != newTip) 
                {
                    pTarget->setData(newTip, ROLE_VPARAM_TOOLTIP);
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
                    if (ctrl == CONTROL_GROUP_LINE) 
                    {
                        addControlGroup(bSubInput, name, ctrl);
                        QStandardItem *pNewItem = pCurrent->clone();
                        appliedItem->appendRow(pNewItem);
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
                        params["defl"].controlProps = pCurrent->data(ROLE_VPARAM_CTRL_PROPERTIES);
                        node[ROLE_PARAMETERS] = QVariant::fromValue(params);

                        m_pGraphsModel->addNode(node, subgIdx, true);

                        //the newItem is created just now, after adding the subgraph node.
                        int dstRow = 0;
                        VParamItem* newItem = pTargetGroup->getItem(name, &dstRow);
                        ZASSERT_EXIT(newItem);
                        pCurrent->m_uuid = newItem->m_uuid;

                        //move the new item to the r-th position.
                        QModelIndex parent = pTargetGroup->index();
                        m_model->moveRow(parent, dstRow, parent, r);
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
                        const QString ident = subInOutput.data(ROLE_OBJID).toString();

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
}

void ZEditParamLayoutDlg::onApply()
{
    m_pGraphsModel->beginTransaction("edit custom param for node");
    zeno::scope_exit scope([=]() { m_pGraphsModel->endTransaction(); });
    applyForItem(m_proxyModel->invisibleRootItem(), m_model->invisibleRootItem());
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
