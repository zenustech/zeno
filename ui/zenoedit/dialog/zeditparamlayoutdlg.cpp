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


static CONTROL_ITEM_INFO controlList[] = {
    {"Contrl Group", CONTROL_GROUP_LINE, ""},
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
    {"DoubleSpinBox", CONTROL_HDOUBLESPINBOX, "float"},
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

    QStringList lstCtrls = {
        "Tab",
        "Group"
    };
    for (int i = 0; i < sizeof(controlList) / sizeof(CONTROL_ITEM_INFO); i++)
    {
        if (bNodeUI && (controlList[i].ctrl == CONTROL_HSLIDER || controlList[i].ctrl == CONTROL_SPINBOX_SLIDER))
            continue;
        else if (!bNodeUI && controlList[i].ctrl == CONTROL_GROUP_LINE)
            continue;
        lstCtrls.append(controlList[i].name);
        m_ui->cbControl->addItem(controlList[i].name);
    }

    m_ui->listConctrl->addItems(lstCtrls);
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

    m_ui->paramsView->setModel(m_proxyModel);
    m_ui->paramsView->setItemDelegate(new ParamTreeItemDelegate(m_proxyModel, m_ui->paramsView));

    QItemSelectionModel* selModel = m_ui->paramsView->selectionModel();
    QModelIndex selIdx = selModel->currentIndex();
    const QModelIndex& wtfIdx = m_proxyModel->index(0, 0);
    selModel->setCurrentIndex(wtfIdx, QItemSelectionModel::SelectCurrent);
    m_ui->paramsView->expandAll();

    connect(selModel, SIGNAL(currentChanged(const QModelIndex&, const QModelIndex&)),
            this, SLOT(onTreeCurrentChanged(const QModelIndex&, const QModelIndex&)));
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
    //record this action first.
    const QString& objPath = index.data(ROLE_OBJPATH).toString();
    //ViewParamSetDataCommand *pCommand =
    //    new ViewParamSetDataCommand(m_pGraphsModel, objPath, newValue, ROLE_VPARAM_CTRL_PROPERTIES);
    //m_commandSeq.append(pCommand);
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

        //QString parentPath = idx.parent().data(ROLE_OBJPATH).toString();
        //ViewParamRemoveCommand *pCommand = new ViewParamRemoveCommand(m_pGraphsModel, parentPath, idx.row());
        //m_commandSeq.append(pCommand);
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
    }
    else if (type == VPARAM_PARAM)
    {
        const QString& paramName = name;
        PARAM_CONTROL ctrl = pCurrentItem->m_ctrl;
        const QString& ctrlName = getControl(ctrl).name;
        const QString& dataType = pCurrentItem->m_type;
        QVariant deflVal = pCurrentItem->m_value;
        bool bCoreParam = pCurrentItem->data(ROLE_VPARAM_IS_COREPARAM).toBool();
        QVariant controlProperties = pCurrentItem->data(ROLE_VPARAM_CTRL_PROPERTIES);

        {
            BlockSignalScope scope(m_ui->cbControl);
            BlockSignalScope scope2(m_ui->cbTypes);

            m_ui->cbControl->setEnabled(true);
            m_ui->cbControl->clear();
            QStringList items = UiHelper::getControlLists(dataType, m_bNodeUI);
            m_ui->cbControl->addItems(items);
            m_ui->cbControl->setCurrentText(ctrlName);

            m_ui->cbTypes->setCurrentText(dataType);
            m_ui->cbTypes->setEnabled(bEditable && (!bCoreParam || m_pGraphsModel->IsSubGraphNode(m_nodeIdx)));
        }

        CallbackCollection cbSets;
        cbSets.cbEditFinished = [=](QVariant newValue) {
            proxyModelSetData(pCurrentItem->index(), newValue, ROLE_PARAM_VALUE);
        };
        if (!deflVal.isValid())
            deflVal = UiHelper::initVariantByControl(ctrl);

        QWidget* valueControl = zenoui::createWidget(deflVal, ctrl, dataType, cbSets, controlProperties);
        if (valueControl)
        {
            valueControl->setEnabled(m_pGraphsModel->IsSubGraphNode(m_nodeIdx));
            m_ui->gridLayout->addWidget(valueControl, rowValueControl, 1);
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
        //ViewParamAddCommand *pCmd = new ViewParamAddCommand(m_pGraphsModel, objPath, vParam);
        //m_commandSeq.append(pCmd);
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
        //ViewParamAddCommand *pCmd = new ViewParamAddCommand(m_pGraphsModel, objPath, vParam);
        //m_commandSeq.append(pCmd);
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
        pNewItem->m_ctrl = ctrl.ctrl;
        pNewItem->m_type = ctrl.defaultType;
        pNewItem->m_value = UiHelper::initVariantByControl(ctrl.ctrl);
        
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
                pNewItem->m_sockProp = SOCKPROP_GROUP;
                break;
            }
        }

        pItem->appendRow(pNewItem);

        //record some command about SubInput/SubOutput.
#if 0
        QString objPath = pItem->data(ROLE_OBJPATH).toString();
        VPARAM_INFO vParam = pNewItem->exportParamInfo();
        ViewParamAddCommand *pCmd = new ViewParamAddCommand(m_pGraphsModel, objPath, vParam);
        m_commandSeq.append(pCmd);
#endif
    }
}

void ZEditParamLayoutDlg::recordSubInputCommands(bool bSubInput, VParamItem* pItem)
{
#if 0
    if (!m_bSubgraphNode)
        return;

    const QString& subgName = m_nodeIdx.data(ROLE_OBJNAME).toString();
    const QModelIndex& subgIdx = m_pGraphsModel->index(subgName);

    const QString& vName = pItem->m_name;
    QString coreName;
    if (pItem->data(ROLE_VPARAM_IS_COREPARAM).toBool())
        coreName = pItem->data(ROLE_PARAM_NAME).toString();
    const QString& typeDesc = pItem->m_type;
    const QVariant& deflVal = pItem->m_value;
    const PARAM_CONTROL ctrl = (PARAM_CONTROL)pItem->data(ROLE_PARAM_CTRL).toInt();

    //new added param.
    const QVariant& defl = pItem->data(ROLE_PARAM_VALUE);

    QPointF pos(0, 0);
    //todo: node arrangement.

    //QString subIO_ident = NodesMgr::createNewNode(m_pGraphsModel, subgIdx, bSubInput ? "SubInput" : "SubOutput", pos);
    NODE_DATA newNodeData = NodesMgr::newNodeData(m_pGraphsModel, bSubInput ? "SubInput" : "SubOutput", pos);
    QString subIO_ident = newNodeData[ROLE_OBJID].toString();
    AddNodeCommand *pNewNodeCmd = new AddNodeCommand(subIO_ident, newNodeData, m_pGraphsModel, subgIdx);
    m_commandSeq.append(pNewNodeCmd);

    QString m_namePath, m_typePath, m_deflPath; //todo: fill the target path for SubInput/SubOutput.


    ViewParamSetDataCommand* pNameCmd = new ViewParamSetDataCommand(m_pGraphsModel, m_namePath, vName, ROLE_PARAM_VALUE);
    ViewParamSetDataCommand* pTypeCmd = new ViewParamSetDataCommand(m_pGraphsModel, m_typePath, typeDesc, ROLE_PARAM_TYPE);
    ViewParamSetDataCommand* pDeflTypeCmd = new ViewParamSetDataCommand(m_pGraphsModel, m_deflPath, typeDesc, ROLE_PARAM_TYPE);
    ViewParamSetDataCommand* pDeflValueCmd = new ViewParamSetDataCommand(m_pGraphsModel, m_deflPath, defl, ROLE_PARAM_VALUE);

    //have to bind new param idx on subgraph node.
    IParamModel* _subgnode_paramModel =
        QVariantPtr<IParamModel>::asPtr(m_nodeIdx.data(bSubInput ? ROLE_INPUT_MODEL : ROLE_OUTPUT_MODEL));
    ZASSERT_EXIT(_subgnode_paramModel);
    const QModelIndex& newParamFromItem = _subgnode_paramModel->index(vName);

    const QString& itemObjPath = pItem->data(ROLE_OBJPATH).toString();
    const QString& targetPath = newParamFromItem.data(ROLE_OBJPATH).toString();

    MapParamIndexCommand *pMappingCmd = new MapParamIndexCommand(m_pGraphsModel, itemObjPath, targetPath);

    updateSubgParamControl(m_pGraphsModel, subgName, bSubInput, vName, ctrl);
#endif
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
    info.sockProp = SOCKPROP_GROUP;
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
                                SOCKPROP_NORMAL, "");
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
    QWidget *valueControl = zenoui::createWidget(value, ctrl, dataType, cbSets, controlProperties);
    if (valueControl) {
        valueControl->setEnabled(m_pGraphsModel->IsSubGraphNode(m_nodeIdx));
        m_ui->gridLayout->addWidget(valueControl, rowValueControl, 1);
        VParamItem *pItem = static_cast<VParamItem *>(m_proxyModel->itemFromIndex(layerIdx));
        switchStackProperties(ctrl, pItem);
    }
}

void ZEditParamLayoutDlg::onTypeItemChanged(int idx)
{
    const QString& dataType = m_ui->cbTypes->itemText(idx);
    QModelIndex layerIdx = m_ui->paramsView->currentIndex();
    if (!layerIdx.isValid() && layerIdx.data(ROLE_VPARAM_TYPE) != VPARAM_PARAM)
        return;

    VParamItem* pItem = static_cast<VParamItem*>(m_proxyModel->itemFromIndex(layerIdx));
    pItem->m_type = dataType;
    pItem->m_ctrl = UiHelper::getControlByType(dataType);
    pItem->m_value = UiHelper::initVariantByControl(pItem->m_ctrl);

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
    QWidget *valueControl = zenoui::createWidget(pItem->m_value, pItem->m_ctrl, dataType, cbSets, controlProperties);
    if (valueControl) {
        valueControl->setEnabled(m_pGraphsModel->IsSubGraphNode(m_nodeIdx));
        m_ui->gridLayout->addWidget(valueControl, rowValueControl, 1);
        switchStackProperties(pItem->m_ctrl, pItem);
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
        const QString name = pCurrent->m_name;
        const QString typeDesc = pCurrent->m_type;
        const QVariant value = pCurrent->m_value;
        const PARAM_CONTROL ctrl = pCurrent->m_ctrl;
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
                QModelIndex parent = appliedItem->index();
                bool ret = m_model->moveRow(parent, targetRow, parent, r);
                ZASSERT_EXIT(ret);
                //reacquire pTarget, because the implementation of moveRow is simplily
                //copy data, rather than insert/remove.
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
                if (pTarget->m_type != typeDesc)
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
                if (pTarget->m_value != value)
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
                        const QVariant &defl = pCurrent->m_value;
                        const QString &typeDesc = pCurrent->m_type;
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
                        VParamItem *newItem = pTargetGroup->getItem(name, &dstRow);
                        pCurrent->m_uuid = newItem->m_uuid;
                        ZASSERT_EXIT(newItem);

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
