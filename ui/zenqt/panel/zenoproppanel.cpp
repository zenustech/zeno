#include "zenoproppanel.h"
#include "zenoapplication.h"
#include "model/graphsmanager.h"
#include "model/curvemodel.h"
#include "model/parammodel.h"
#include "variantptr.h"
#include "widgets/zcombobox.h"
#include "widgets/zlabel.h"
#include "style/zenostyle.h"
#include "nodeeditor/gv/zenoparamwidget.h"
#include "widgets/zveceditor.h"
#include "util/uihelper.h"
#include "widgets/zexpandablesection.h"
#include "widgets/zlinewidget.h"
#include "widgets/zlineedit.h"
#include "widgets/ztextedit.h"
#include "widgets/zwidgetfactory.h"
#include "util/log.h"
#include "util/apphelper.h"
#include "util/curveutil.h"
#include "curvemap/zcurvemapeditor.h"
#include "dialog/zenoheatmapeditor.h"
#include "zenomainwindow.h"
#include "dialog/zeditparamlayoutdlg.h"
#include "widgets/zspinboxslider.h"
#include "zenoblackboardpropwidget.h"
#include "widgets/ztimeline.h"
#include "util/apphelper.h"
#include "zassert.h"
#include "widgets/zcheckbox.h"
#include "ZenoDictListLinksPanel.h"
#include "zenoDopNetworkPanel.h"


using namespace zeno::reflect;

class RetryScope
{
public:
    RetryScope(bool& bRetry)
        : m_bRetry(bRetry)
    {
        m_bRetry = true;
    }
    ~RetryScope()
    {
        m_bRetry = false;
    }
private:
    bool& m_bRetry;
};


ZenoPropPanel::ZenoPropPanel(QWidget* parent)
    : QWidget(parent)
    , m_bReentry(false)
    , m_tabWidget(nullptr)
    , m_normalNodeInputWidget(nullptr)
    , m_outputWidget(nullptr)
    , m_dictListLinksTable(nullptr)
    , m_dopNetworkPanel(nullptr)
    , m_hintlist(new ZenoHintListWidget)
    , m_descLabel(new ZenoFuncDescriptionLabel)
{
    QVBoxLayout* pVLayout = new QVBoxLayout;
    pVLayout->setContentsMargins(QMargins(0, 0, 0, 0));
    setLayout(pVLayout);
    setFocusPolicy(Qt::ClickFocus);
}

ZenoPropPanel::~ZenoPropPanel()
{
}

QSize ZenoPropPanel::sizeHint() const
{
    QSize sz = QWidget::sizeHint();
    return sz;
}

QSize ZenoPropPanel::minimumSizeHint() const
{
    QSize sz = QWidget::minimumSizeHint();
    return sz;
}

bool ZenoPropPanel::updateCustomName(const QString &value, QString &oldValue) 
{
    if (!m_idx.isValid())
        return false;

    oldValue = m_idx.data(ROLE_NODE_NAME).toString();
    if (value == oldValue)
        return true;

    if (GraphModel* pModel = QVariantPtr<GraphModel>::asPtr(m_idx.data(ROLE_GRAPH)))
    {
        QString name = pModel->updateNodeName(m_idx, value);
        if (name != value)
        {
            QMessageBox::warning(nullptr, tr("Rename warring"), tr("The name %1 is existed").arg(value));
            return false;
        }
    }
    return true;
}

ZenoHintListWidget* ZenoPropPanel::getHintListInstance()
{
    return m_hintlist.get();
}

ZenoFuncDescriptionLabel* ZenoPropPanel::getFuncDescriptionInstance()
{
    return m_descLabel.get();
}

void ZenoPropPanel::clearLayout()
{
    m_hintlist->setParent(nullptr);
    m_descLabel->setParent(nullptr);

    setUpdatesEnabled(false);
    qDeleteAll(findChildren<QWidget*>(QString(), Qt::FindDirectChildrenOnly));
    QVBoxLayout* pMainLayout = qobject_cast<QVBoxLayout*>(this->layout());
    while (pMainLayout->count() > 0)
    {
        QLayoutItem* pItem = pMainLayout->itemAt(pMainLayout->count() - 1);
        pMainLayout->removeItem(pItem);
    }
    setUpdatesEnabled(true);

    if (m_idx.data(ROLE_CLASS_NAME).toString() == "MakeDict" || m_idx.data(ROLE_CLASS_NAME).toString() == "MakeList") {
        clearMakeDictMakeListLayout();
            }
    else {
        int nodeType = m_idx.data(ROLE_NODETYPE).toInt();
        if (nodeType == zeno::Node_SubgraphNode || nodeType == zeno::Node_AssetInstance) {
            QString clsname = m_idx.data(ROLE_CLASS_NAME).toString();
            if (clsname == "Subnet") {
                m_tabWidget = nullptr;
                m_outputWidget = nullptr;
        }
            else if (clsname == "DopNetwork") {
                m_dopNetworkPanel = nullptr;
                m_outputWidget = nullptr;
    }
        } else {
        m_normalNodeInputWidget = nullptr;
        m_outputWidget = nullptr;
        }
        m_inputControls.clear();
        m_outputControls.clear();
        m_floatControls.clear();

        if (m_idx.isValid())
        {
            QStandardItemModel* paramsModel = QVariantPtr<ParamsModel>::asPtr(m_idx.data(ROLE_PARAMS))->customParamModel();
            if (paramsModel)
            {
                disconnect(paramsModel, &QStandardItemModel::rowsInserted, this, &ZenoPropPanel::onViewParamInserted);
                disconnect(paramsModel, &QStandardItemModel::rowsAboutToBeRemoved, this, &ZenoPropPanel::onViewParamAboutToBeRemoved);
                disconnect(paramsModel, &QStandardItemModel::dataChanged, this, &ZenoPropPanel::onCustomParamDataChanged);
                disconnect(paramsModel, &QStandardItemModel::rowsMoved, this, &ZenoPropPanel::onViewParamsMoved);
            }
        }
    }

    update();
}

void ZenoPropPanel::reset(GraphModel* subgraph, const QModelIndexList& nodes, bool select)
{
    if (m_bReentry)
        return;

    RetryScope scope(m_bReentry);

    if (!select || nodes.size() != 1 || m_idx == nodes[0])
    {
        //update();
        return;
    }

    clearLayout();
    QVBoxLayout* pMainLayout = qobject_cast<QVBoxLayout*>(this->layout());

    m_model = subgraph;
    m_idx = nodes[0];
    if (!m_idx.isValid())
        return;

    if (m_idx.data(ROLE_CLASS_NAME).toString() == "MakeDict" || m_idx.data(ROLE_CLASS_NAME).toString() == "MakeList") {
        if (QWidget* wid = resetMakeDictMakeListLayout()) {
            pMainLayout->addWidget(wid);
            }
        }
    else {
        ParamsModel* paramsM = QVariantPtr<ParamsModel>::asPtr(m_idx.data(ROLE_PARAMS));
        ZASSERT_EXIT(paramsM);
        QStandardItemModel* treeModel = paramsM->customParamModel();
        if (!treeModel)
            return;

        connect(treeModel, &QStandardItemModel::rowsInserted, this, &ZenoPropPanel::onViewParamInserted);
        connect(treeModel, &QStandardItemModel::rowsAboutToBeRemoved, this, &ZenoPropPanel::onViewParamAboutToBeRemoved);
        connect(treeModel, &QStandardItemModel::dataChanged, this, &ZenoPropPanel::onCustomParamDataChanged);
        connect(treeModel, &QStandardItemModel::rowsMoved, this, &ZenoPropPanel::onViewParamsMoved);
        connect(treeModel, &QStandardItemModel::modelAboutToBeReset, this, [=]() {
            //clear all
            m_inputControls.clear();
        m_outputControls.clear();
        m_floatControls.clear();
        if (m_tabWidget)
        {
            while (m_tabWidget->count() > 0)
            {
                QWidget *wid = m_tabWidget->widget(0);
                m_tabWidget->removeTab(0);
                delete wid;
            }
        }
        if (m_normalNodeInputWidget) {
            m_normalNodeInputWidget->deleteLater();
            m_normalNodeInputWidget = nullptr;
        }
        if (m_outputWidget) {
            m_outputWidget->deleteLater();
            m_outputWidget = nullptr;
        }
            });

        connect(m_model, &GraphModel::nodeRemoved, this, &ZenoPropPanel::onNodeRemoved, Qt::UniqueConnection);
        connect(paramsM, &ParamsModel::enabledVisibleChanged, this, &ZenoPropPanel::onUpdateParamsVisbleEnabled);

        QStandardItem* root = treeModel->invisibleRootItem();
        if (!root) return;

        QStandardItem* pInputs = root->child(0);
        if (!pInputs) return;

        QStandardItem* pOutputs = root->child(1);
        if (!pOutputs) return;

        int nodeType = m_idx.data(ROLE_NODETYPE).toInt();
        if (nodeType == zeno::Node_SubgraphNode || nodeType == zeno::Node_AssetInstance) {
            //子图节点布局
            QString clsname = m_idx.data(ROLE_CLASS_NAME).toString();
            if (clsname == "DopNetwork") {
                if (QWidget* wid = resetDopNetworkLayout()) {
                    pMainLayout->addWidget(wid);
                }
            } else {//subnet和asset
                if (QWidget* wid = resetSubnetLayout()) {
                    pMainLayout->addWidget(wid);
                }
            }
        } else {
            //普通节点布局
            if (QWidget* wid = resetNormalNodeLayout()) {
                pMainLayout->addWidget(wid);
        }
    }
    }
}

void ZenoPropPanel::onViewParamInserted(const QModelIndex& parent, int first, int last)
{
    QStandardItemModel* paramsModel = QVariantPtr<ParamsModel>::asPtr(m_idx.data(ROLE_PARAMS))->customParamModel();
    ZASSERT_EXIT(paramsModel);

    QStandardItem* root = paramsModel->invisibleRootItem();
    ZASSERT_EXIT(root);

    QStandardItem* parentItem = paramsModel->itemFromIndex(parent);
    ZASSERT_EXIT(parentItem);

    QStandardItem* newItem = parentItem->child(first);
    if (newItem->data(ROLE_PARAM_GROUP) == zeno::Role_InputPrimitive) {
        if (!m_idx.isValid())
            return;
        //subnet节点，可能有多个tab
        if ((m_idx.data(ROLE_NODETYPE) == zeno::Node_SubgraphNode || m_idx.data(ROLE_NODETYPE) == zeno::Node_AssetInstance)) {
            ZASSERT_EXIT(m_tabWidget);

            if (m_inputControls.isEmpty()) {
                return;
            }

            VPARAM_TYPE vType = (VPARAM_TYPE)newItem->data(ROLE_ELEMENT_TYPE).toInt();
            const QString& name = newItem->data(ROLE_PARAM_NAME).toString();
            if (vType == VPARAM_TAB)
            {
                syncAddTab(m_tabWidget, newItem, first);
            }
            else if (vType == VPARAM_GROUP)
            {
                ZASSERT_EXIT(parentItem->data(ROLE_ELEMENT_TYPE) == VPARAM_TAB);
                const QString& tabName = parentItem->data(ROLE_PARAM_NAME).toString();
                int idx = UiHelper::tabIndexOfName(m_tabWidget, tabName);
                QWidget* tabWid = m_tabWidget->widget(idx);
                QVBoxLayout* pTabLayout = qobject_cast<QVBoxLayout*>(tabWid->layout());
                if (pTabLayout == nullptr)
                {
                    pTabLayout = new QVBoxLayout;
                    pTabLayout->addStretch();
                    tabWid->setLayout(pTabLayout);
                }
                QLayoutItem *layoutItem = pTabLayout->itemAt(pTabLayout->count() - 1);
                if (layoutItem && dynamic_cast<QSpacerItem *>(layoutItem))
                    pTabLayout->removeItem(layoutItem);
                syncAddGroup(pTabLayout, newItem, first);
                pTabLayout->addStretch();
            }
            else if (vType == VPARAM_PARAM)
            {
                ZASSERT_EXIT(parentItem->data(ROLE_ELEMENT_TYPE) == VPARAM_GROUP);

                QStandardItem* pTabItem = parentItem->parent();
                ZASSERT_EXIT(pTabItem && pTabItem->data(ROLE_ELEMENT_TYPE) == VPARAM_TAB);

                const QString& tabName = pTabItem->data(ROLE_PARAM_NAME).toString();
                const QString& groupName = parentItem->data(ROLE_PARAM_NAME).toString();
                const QString& paramName = name;

                ZExpandableSection* pGroupWidget = findGroup(tabName, groupName);
                if (!pGroupWidget)
                    return;

                QGridLayout* pGroupLayout = qobject_cast<QGridLayout*>(pGroupWidget->contentLayout());
                ZASSERT_EXIT(pGroupLayout);
                if (pGroupWidget->title() == groupName)
                {
                    QStandardItem* paramItem = parentItem->child(first);
                    bool ret = syncAddControl(pGroupWidget, pGroupLayout, paramItem, first);
                    if (ret)
                    {
                        pGroupWidget->updateGeo();
                    }
                }
            }
        }
        else {  //普通节点，单个tab
            ZASSERT_EXIT(parentItem->data(ROLE_ELEMENT_TYPE) == VPARAM_GROUP);
            QStandardItem* pTabItem = parentItem->parent();
            ZASSERT_EXIT(pTabItem && pTabItem->data(ROLE_ELEMENT_TYPE) == VPARAM_TAB);

            if (ZScrollArea* area = qobject_cast<ZScrollArea*>(m_normalNodeInputWidget)) {
                QList<QGridLayout*> layouts = area->findChildren<QGridLayout*>();
                if (!layouts.empty()) {
                    normalNodeAddInputWidget(area, layouts[0], parentItem, first);
                }
            }
        }
    }
    else if (newItem->data(ROLE_PARAM_GROUP) == zeno::Role_OutputPrimitive) {
        if (ZScrollArea* area = qobject_cast<ZScrollArea*>(m_outputWidget)) {
            QList<QGridLayout*> layouts = area->findChildren<QGridLayout*>();
            if (!layouts.empty()) {
                addOutputWidget(area, layouts[0], parentItem, first);
            }
            m_outputWidget->setMinimumHeight(40);
        }
    }
    /*
    ViewParamModel *pModel = qobject_cast<ViewParamModel*>(sender());
    if (pModel && !newItem->data(ROLE_VPARAM_IS_COREPARAM).toBool())
        pModel->markDirty();
    */
}

void ZenoPropPanel::onUpdateParamsVisbleEnabled()
{
    bool bUpdate = true;
    for (auto& [_, tab] : m_inputControls) {
        for (auto& [_, group] : tab) {
            for (auto& [_, control] : group) {
                bool bEnable = control.m_coreparamIdx.data(ROLE_PARAM_ENABLE).toBool();
                bool bVisible = control.m_coreparamIdx.data(ROLE_PARAM_VISIBLE).toBool();

                control.pControl->setVisible(bVisible);
                control.pIconLabel->setVisible(bVisible);
                control.pLabel->setVisible(bVisible);

                control.pControl->setEnabled(bEnable);
                control.pIconLabel->setEnabled(bEnable);
                control.pLabel->setEnabled(bEnable);
            }
        }
    }
    if (bUpdate)
        update();
}

QWidget* ZenoPropPanel::resetMakeDictMakeListLayout()
{
    if (ParamsModel* paramsModel = QVariantPtr<ParamsModel>::asPtr(m_idx.data(ROLE_PARAMS))) {
        connect(paramsModel, &ParamsModel::linkAboutToBeRemoved, this, &ZenoPropPanel::onLinkRemoved);
        connect(paramsModel, &ParamsModel::linkAboutToBeInserted, this, &ZenoPropPanel::onLinkAdded);
        QModelIndex inputObjsIdx = paramsModel->paramIdx("objs", true);
        if (inputObjsIdx.isValid()) {
            m_dictListLinksTable = new ZenoDictListLinksTable(2, this);
            DragDropModel* dragdropModel = new DragDropModel(inputObjsIdx, 2, m_dictListLinksTable);
            m_dictListLinksTable->setModel(dragdropModel);
            m_dictListLinksTable->init();
            connect(m_dictListLinksTable, &ZenoDictListLinksTable::linksUpdated, this, &ZenoPropPanel::onDictListTableUpdateLink);
            connect(m_dictListLinksTable, &ZenoDictListLinksTable::linksRemoved, this, &ZenoPropPanel::onDictListTableRemoveLink);
            return m_dictListLinksTable;
        }
    }
    return nullptr;
}

void ZenoPropPanel::clearMakeDictMakeListLayout()
{
    m_dictListLinksTable = nullptr;

    if (m_idx.isValid()) {
        if (ParamsModel* paramsModel = QVariantPtr<ParamsModel>::asPtr(m_idx.data(ROLE_PARAMS))) {
            disconnect(paramsModel, &ParamsModel::linkAboutToBeRemoved, this, &ZenoPropPanel::onLinkRemoved);
            disconnect(paramsModel, &ParamsModel::linkAboutToBeInserted, this, &ZenoPropPanel::onLinkAdded);
            disconnect(m_dictListLinksTable, &ZenoDictListLinksTable::linksUpdated, this, &ZenoPropPanel::onDictListTableUpdateLink);
            disconnect(m_dictListLinksTable, &ZenoDictListLinksTable::linksRemoved, this, &ZenoPropPanel::onDictListTableRemoveLink);
        }
    }
}

void ZenoPropPanel::clearDopNetworkLayout() {

}

QWidget* ZenoPropPanel::resetDopNetworkLayout()
{
    if (QWidget* wid = resetSubnetLayout()) {
        m_dopNetworkPanel = new zenoDopNetworkPanel(wid, this, m_idx);
        m_dopNetworkPanel->tabBar()->setProperty("cssClass", "propanel");
        m_dopNetworkPanel->setDocumentMode(true);
        m_dopNetworkPanel->setTabsClosable(false);
        m_dopNetworkPanel->setMovable(false);
        QFont font = QApplication::font();
        font.setWeight(QFont::Medium);
        m_dopNetworkPanel->setFont(font); //bug in qss font setting.
        m_dopNetworkPanel->tabBar()->setDrawBase(false);
        return m_dopNetworkPanel;
    }
    return nullptr;
}

QWidget* ZenoPropPanel::resetSubnetLayout()
{
    QStandardItemModel* paramsModel = QVariantPtr<ParamsModel>::asPtr(m_idx.data(ROLE_PARAMS))->customParamModel();
    if (!paramsModel)
        return nullptr;
    QStandardItem* root = paramsModel->invisibleRootItem();
    if (!root)
        return nullptr;
    QStandardItem* pInputs = root->child(0);
    if (!pInputs)
        return nullptr;

    QSplitter* splitter = new QSplitter(Qt::Vertical, this);
    splitter->setStyleSheet("QSplitter::handle {" "background-color: rgb(0,0,0);" "height: 2px;" "}");

    m_tabWidget = new QTabWidget(this);
    m_tabWidget->tabBar()->setProperty("cssClass", "propanel");
    m_tabWidget->setDocumentMode(true);
    m_tabWidget->setTabsClosable(false);
    m_tabWidget->setMovable(false);

    QFont font = QApplication::font();
    font.setWeight(QFont::Medium);

    m_tabWidget->setFont(font); //bug in qss font setting.
    m_tabWidget->tabBar()->setDrawBase(false);

    for (int i = 0; i < pInputs->rowCount(); i++)
    {
        QStandardItem* pTabItem = pInputs->child(i);
        syncAddTab(m_tabWidget, pTabItem, i);
    }
    m_hintlist->setParent(m_tabWidget);
    m_hintlist->resetSize();
    m_hintlist->setCalcPropPanelPosFunc([this]() -> QPoint {return m_tabWidget->mapToGlobal(QPoint(0, 0)); });
    m_descLabel->setParent(m_tabWidget);

    splitter->addWidget(m_tabWidget);
    update();
    if (QWidget* outputsWidget = resetOutputs()) {
        splitter->addWidget(outputsWidget);
    }
    splitter->setStretchFactor(0, 3);
    splitter->setStretchFactor(1, 1);
    return splitter;
}

bool ZenoPropPanel::syncAddControl(ZExpandableSection* pGroupWidget, QGridLayout* pGroupLayout, QStandardItem* paramItem, int row)
{
    ZASSERT_EXIT(paramItem && pGroupLayout, false);
    QStandardItem* pGroupItem = paramItem->parent();
    ZASSERT_EXIT(pGroupItem, false);
    QStandardItem* pTabItem = pGroupItem->parent();
    ZASSERT_EXIT(pTabItem, false);

    const QString& tabName = pTabItem->data(ROLE_PARAM_NAME).toString();
    const QString& groupName = pGroupItem->data(ROLE_PARAM_NAME).toString();
    const QString& paramName = paramItem->data(ROLE_PARAM_NAME).toString();
    const zeno::reflect::Any& anyVal = paramItem->data(ROLE_PARAM_VALUE).value<zeno::reflect::Any>();
    //QVariant val = UiHelper::anyToQvar();
    zeno::ParamControl ctrl = (zeno::ParamControl)paramItem->data(ROLE_PARAM_CONTROL).toInt();

    const zeno::ParamType type = (zeno::ParamType)paramItem->data(ROLE_PARAM_TYPE).toLongLong();
    const zeno::reflect::Any &pros = paramItem->data(ROLE_PARAM_CTRL_PROPERTIES).value<zeno::reflect::Any>();

    QPersistentModelIndex perIdx(paramItem->index());
    CallbackCollection cbSet;

    if (ctrl == zeno::Seperator)
    {
        return false;
    }

    bool bFloat = UiHelper::isFloatType(type);
    cbSet.cbEditFinished = [=](zeno::reflect::Any newValue) {
        /*
        if (bFloat)
        {
            //if (!AppHelper::updateCurve(paramItem->data(ROLE_PARAM_VALUE), newValue))
            //{
                //onCustomParamDataChanged(perIdx, perIdx, QVector<int>() << ROLE_PARAM_VALUE);
                //return;
            //}
            QStandardItemModel* paramsModel = QVariantPtr<ParamsModel>::asPtr(m_idx.data(ROLE_PARAMS))->customParamModel();
            BlockSignalScope scope(paramsModel); //setData时需屏蔽dataChange信号
            paramsModel->setData(perIdx, newValue, ROLE_PARAM_VALUE);
        }
        */
        ParamsModel* paramsModel = QVariantPtr<ParamsModel>::asPtr(m_idx.data(ROLE_PARAMS));
        const QModelIndex& idx = paramsModel->paramIdx(perIdx.data(ROLE_PARAM_NAME).toString(), true);
        UiHelper::qIndexSetData(idx, QVariant::fromValue(newValue), ROLE_PARAM_VALUE);
    };
    cbSet.cbSwitch = [=](bool bOn) {
        zenoApp->getMainWindow()->setInDlgEventLoop(bOn);   //deal with ubuntu dialog slow problem when update viewport.
    };
    cbSet.cbGetIndexData = [=]() -> QVariant { 
        return perIdx.isValid() ? paramItem->data(ROLE_PARAM_VALUE) : QVariant();
    };
    //key frame
    bool bKeyFrame = false;
    if (bFloat)
    {
        //bKeyFrame = AppHelper::getCurveValue(val);
    }

    QWidget* pControl = zenoui::createWidget(m_idx, anyVal, ctrl, type, cbSet, pros);

    ZTextLabel* pLabel = new ZTextLabel(paramName);

    QFont font = QApplication::font();
    font.setWeight(QFont::Light);
    pLabel->setFont(font);
    pLabel->setToolTip(paramItem->data(ROLE_PARAM_TOOLTIP).toString());

    pLabel->setTextColor(QColor(255, 255, 255, 255 * 0.7));
    pLabel->setHoverCursor(Qt::ArrowCursor);
    //pLabel->setProperty("cssClass", "proppanel");
    bool bVisible = paramItem->data(ROLE_PARAM_SOCKET_VISIBLE).toBool();

    ZIconLabel* pIcon = new ZIconLabel();
    pIcon->setIcons(ZenoStyle::dpiScaledSize(QSize(26, 26)), ":/icons/parameter_key-frame_idle.svg", ":/icons/parameter_key-frame_hover.svg",
        ":/icons/parameter_key-frame_correct.svg", ":/icons/parameter_key-frame_correct.svg");
    pIcon->toggle(bVisible);
    connect(pIcon, &ZIconLabel::toggled, this, [=](bool toggled) {
        ParamsModel* paramsModel = QVariantPtr<ParamsModel>::asPtr(m_idx.data(ROLE_PARAMS));
        const QModelIndex& idx = paramsModel->paramIdx(perIdx.data(ROLE_PARAM_NAME).toString(), true);
        UiHelper::qIndexSetData(idx, toggled, ROLE_PARAM_SOCKET_VISIBLE);
    });
    pGroupLayout->addWidget(pIcon, row, 0, Qt::AlignCenter);

    pGroupLayout->addWidget(pLabel, row, 1, Qt::AlignLeft | Qt::AlignVCenter);
    if (pControl)
        pGroupLayout->addWidget(pControl, row, 2, Qt::AlignVCenter);

    if (ZTextEdit* pMultilineStr = qobject_cast<ZTextEdit*>(pControl))
    {
        connect(pMultilineStr, &ZTextEdit::geometryUpdated, pGroupWidget, &ZExpandableSection::updateGeo);
    } else if (ZLineEdit* pLineEdit = qobject_cast<ZLineEdit*>(pControl)) {
        pLineEdit->setHintListWidget(m_hintlist.get(), m_descLabel.get());
    }
    else if (ZVecEditor* pVecEdit = qobject_cast<ZVecEditor*>(pControl)) {
        pVecEdit->setHintListWidget(m_hintlist.get(), m_descLabel.get());
    }

    _PANEL_CONTROL panelCtrl;
    panelCtrl.controlLayout = pGroupLayout;
    panelCtrl.pLabel = pLabel;
    panelCtrl.pIconLabel = pIcon;
    panelCtrl.m_coreparamIdx = perIdx;
    panelCtrl.pControl = pControl;

    m_inputControls[tabName][groupName][paramName] = panelCtrl;

    if (bFloat && pControl) {
        m_floatControls << panelCtrl;
        pLabel->installEventFilter(this);
        pControl->installEventFilter(this);   
        ZenoMainWindow* mainWin = zenoApp->getMainWindow();
        ZASSERT_EXIT(mainWin, true);
        ZTimeline* timeline = mainWin->timeline();
        ZASSERT_EXIT(timeline, true);
        onUpdateFrame(pControl, timeline->value(), paramItem->data(ROLE_PARAM_VALUE));
        connect(timeline, &ZTimeline::sliderValueChanged, pControl, [=](int nFrame) {
            onUpdateFrame(pControl, nFrame, paramItem->data(ROLE_PARAM_VALUE));
            }, Qt::UniqueConnection);
        connect(mainWin, &ZenoMainWindow::visFrameUpdated, pControl, [=](bool bGLView, int nFrame) {
            onUpdateFrame(pControl, nFrame, paramItem->data(ROLE_PARAM_VALUE));
            }, Qt::UniqueConnection);
    }
    return true;
}

bool ZenoPropPanel::syncAddGroup(QVBoxLayout* pTabLayout, QStandardItem* pGroupItem, int row)
{
    const QString& groupName = pGroupItem->data(Qt::DisplayRole).toString();
    bool bCollaspe = false;//TOOD: collasped.
    ZExpandableSection* pGroupWidget = new ZExpandableSection(groupName);
    pGroupWidget->setObjectName(groupName);
    pGroupWidget->setCollasped(bCollaspe);
    QGridLayout* pLayout = new QGridLayout;
    pLayout->setContentsMargins(10, 15, 10, 15);
    //pLayout->setColumnStretch(1, 1);
    pLayout->setColumnStretch(2, 3);
    pLayout->setSpacing(10);
    for (int k = 0; k < pGroupItem->rowCount(); k++)
    {
        QStandardItem* paramItem = pGroupItem->child(k);
        syncAddControl(pGroupWidget, pLayout, paramItem, k);
    }
    pGroupWidget->setContentLayout(pLayout);
    pTabLayout->addWidget(pGroupWidget);

    connect(pGroupWidget, &ZExpandableSection::stateChanged, this, [=](bool bCollasped) {
        if (!m_idx.isValid())
            return;
        //todo: search groupitem by model, not by the pointer directly.
        //pGroupItem->setData(bCollasped, ROLE_VPARAM_COLLASPED);
    });
    return true;
}

bool ZenoPropPanel::syncAddTab(QTabWidget* pTabWidget, QStandardItem* pTabItem, int row)
{
    const QString& tabName = pTabItem->data(Qt::DisplayRole).toString();
    QWidget* tabWid = pTabWidget->widget(UiHelper::tabIndexOfName(pTabWidget, tabName));
    if (tabWid)
        return false;

    QWidget* pTabWid = new QWidget;
    QVBoxLayout* pTabLayout = new QVBoxLayout;
    pTabLayout->setContentsMargins(QMargins(0, 0, 0, 0));
    pTabLayout->setSpacing(0);
    if (m_idx.data(ROLE_NODETYPE) == zeno::Node_Group) 
    {
        ZenoBlackboardPropWidget *propWidget = new ZenoBlackboardPropWidget(m_idx, pTabWid);
        pTabLayout->addWidget(propWidget);
    } 
    else 
    {
        for (int j = 0; j < pTabItem->rowCount(); j++) {
            QStandardItem* pGroupItem = pTabItem->child(j);
            syncAddGroup(pTabLayout, pGroupItem, j);
        }
    }

    pTabLayout->addStretch();
    pTabWid->setLayout(pTabLayout);
    pTabWidget->insertTab(row, pTabWid, tabName);
    return true;
}

ZExpandableSection* ZenoPropPanel::findGroup(const QString& tabName, const QString& groupName)
{
    QWidget* tabWid = m_tabWidget->widget(UiHelper::tabIndexOfName(m_tabWidget, tabName));
    ZASSERT_EXIT(tabWid, nullptr);
    auto lst = tabWid->findChildren<ZExpandableSection*>(QString(), Qt::FindDirectChildrenOnly);
    for (ZExpandableSection* pGroupWidget : lst)
    {
        if (pGroupWidget->title() == groupName)
        {
            return pGroupWidget;
        }
    }
    return nullptr;
}

QWidget* ZenoPropPanel::resetNormalNodeLayout()
{
    QStandardItemModel* paramsModel = QVariantPtr<ParamsModel>::asPtr(m_idx.data(ROLE_PARAMS))->customParamModel();
    if (!paramsModel)
        return nullptr;

    QStandardItem* root = paramsModel->invisibleRootItem();
    if (!root) return nullptr;

    QStandardItem* pInputs = root->child(0);
    if (!pInputs) return nullptr;

    QSplitter* splitter = new QSplitter(Qt::Vertical, this);
    splitter->setStyleSheet("QSplitter::handle {" "background-color: rgb(0,0,0);" "height: 2px;" "}");

    ZScrollArea* scrollArea = new ZScrollArea(this);
    scrollArea->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    scrollArea->setMinimumHeight(0);
    scrollArea->setFrameShape(QFrame::NoFrame);
    scrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    scrollArea->setWidgetResizable(true);
    ZContentWidget* pWidget = new ZContentWidget(scrollArea);
    pWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    QGridLayout* pLayout = new QGridLayout(pWidget);
    scrollArea->setWidget(pWidget);
    pLayout->setContentsMargins(10, 15, 10, 15);
    pLayout->setAlignment(Qt::AlignTop);
    pLayout->setColumnStretch(2, 3);
    pLayout->setSpacing(10);
    m_normalNodeInputWidget = scrollArea;

    QStandardItem* pGroupItem = pInputs->child(0)->child(0);
    for (int row = 0; row < pGroupItem->rowCount(); row++) {
        normalNodeAddInputWidget(scrollArea, pLayout, pGroupItem, row);
    }

    m_hintlist->setParent(m_normalNodeInputWidget);
    m_hintlist->resetSize();
    m_hintlist->setCalcPropPanelPosFunc([this]() -> QPoint {return m_normalNodeInputWidget->mapToGlobal(QPoint(0, 0)); });
    m_descLabel->setParent(m_normalNodeInputWidget);

    splitter->addWidget(m_normalNodeInputWidget);
    if (QWidget* outputsWidget = resetOutputs()) {
        splitter->addWidget(outputsWidget);
    }
    splitter->setStretchFactor(0, 3);
    splitter->setStretchFactor(1, 1);
    return splitter;
}

QWidget* ZenoPropPanel::resetOutputs()
{
    QStandardItemModel* paramsModel = QVariantPtr<ParamsModel>::asPtr(m_idx.data(ROLE_PARAMS))->customParamModel();
    if (!paramsModel)
        return nullptr;

    QStandardItem* root = paramsModel->invisibleRootItem();
    if (!root) return nullptr;

    QStandardItem* pOutputs = root->child(1);
    if (!pOutputs) return nullptr;

    ZScrollArea* scrollArea = new ZScrollArea(this);
    scrollArea->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    scrollArea->setMinimumHeight(0);
    scrollArea->setFrameShape(QFrame::NoFrame);
    scrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    scrollArea->setWidgetResizable(true);
    ZContentWidget* pWidget = new ZContentWidget(scrollArea);
    pWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    QGridLayout* pLayout = new QGridLayout(pWidget);
    scrollArea->setWidget(pWidget);
    pLayout->setContentsMargins(10, 15, 10, 15);
    pLayout->setAlignment(Qt::AlignTop);
    pLayout->setColumnStretch(1, 3);
    pLayout->setSpacing(10);

    for (int row = 0; row < pOutputs->rowCount(); row++) {
        addOutputWidget(scrollArea, pLayout, pOutputs, row);
    }
    m_outputWidget = scrollArea;
    return m_outputWidget;
}

void ZenoPropPanel::normalNodeAddInputWidget(ZScrollArea* scrollArea, QGridLayout* pLayout, QStandardItem* pGroupItem, int row)
{
    ZASSERT_EXIT(pGroupItem);
    QStandardItem* pTabItem = pGroupItem->parent();
    ZASSERT_EXIT(pTabItem);

    const QString& tabName = pTabItem->data(ROLE_PARAM_NAME).toString();
    const QString& groupName = pGroupItem->data(ROLE_PARAM_NAME).toString();

    auto paramItem = pGroupItem->child(row);
    QModelIndex paramIdx = paramItem->index();
    const QString& paramName = paramItem->data(ROLE_PARAM_NAME).toString();

    zeno::reflect::Any anyVal = paramItem->data(ROLE_PARAM_VALUE).value<zeno::reflect::Any>();
    const zeno::ParamType type = (zeno::ParamType)paramItem->data(ROLE_PARAM_TYPE).toLongLong();

    bool bInput = paramItem->data(ROLE_ISINPUT).toBool();
    ParamsModel* paramsM = QVariantPtr<ParamsModel>::asPtr(m_idx.data(ROLE_PARAMS));
    QPersistentModelIndex idxCoreParam = paramsM->paramIdx(paramName, bInput);

    ZASSERT_EXIT(type == Param_Wildcard || anyVal.has_value());

    zeno::ParamControl ctrl = (zeno::ParamControl)idxCoreParam.data(ROLE_PARAM_CONTROL).toInt();

    const zeno::reflect::Any& pros = idxCoreParam.data(ROLE_PARAM_CTRL_PROPERTIES).value<zeno::reflect::Any>();

    CallbackCollection cbSet;

    bool bFloat = UiHelper::isFloatType(type);
    cbSet.cbEditFinished = [=](zeno::reflect::Any newValue) {
        paramItem->setData(QVariant::fromValue(newValue), ROLE_PARAM_VALUE);
    };
    cbSet.cbSwitch = [=](bool bOn) {
        zenoApp->getMainWindow()->setInDlgEventLoop(bOn);   //deal with ubuntu dialog slow problem when update viewport.
    };
    cbSet.cbGetIndexData = [=]() -> QVariant {
        return idxCoreParam.isValid() ? idxCoreParam.data(ROLE_PARAM_VALUE) : QVariant();
    };

    //key frame
    bool bKeyFrame = false;
    if (bFloat)
    {
        //bKeyFrame = AppHelper::getCurveValue(val);
    }

    QWidget* pControl = zenoui::createWidget(m_idx, anyVal, ctrl, type, cbSet, pros);

    ZTextLabel* pLabel = new ZTextLabel(paramName);

    QFont font = QApplication::font();
    font.setWeight(QFont::Light);
    pLabel->setFont(font);
    pLabel->setToolTip(idxCoreParam.data(ROLE_PARAM_TOOLTIP).toString());

    pLabel->setTextColor(QColor(255, 255, 255, 255 * 0.7));
    pLabel->setHoverCursor(Qt::ArrowCursor);
    //pLabel->setProperty("cssClass", "proppanel");

    bool bVisible = idxCoreParam.data(ROLE_PARAM_VISIBLE).toBool();
    bool bEnable = idxCoreParam.data(ROLE_PARAM_ENABLE).toBool();
    bool bSocketVisible = idxCoreParam.data(ROLE_PARAM_SOCKET_VISIBLE).toBool();
    ZIconLabel* pIcon = new ZIconLabel();
    pIcon->setIcons(ZenoStyle::dpiScaledSize(QSize(26, 26)), ":/icons/parameter_key-frame_idle.svg", ":/icons/parameter_key-frame_hover.svg",
        ":/icons/parameter_key-frame_correct.svg", ":/icons/parameter_key-frame_correct.svg");
    pIcon->toggle(bSocketVisible);
    connect(pIcon, &ZIconLabel::toggled, this, [=](bool toggled) {
        UiHelper::qIndexSetData(idxCoreParam, toggled, ROLE_PARAM_SOCKET_VISIBLE);
        });
    pLayout->addWidget(pIcon, row, 0, Qt::AlignCenter);

    pLayout->addWidget(pLabel, row, 1, Qt::AlignLeft | Qt::AlignVCenter);
    if (pControl)
        pLayout->addWidget(pControl, row, 2, Qt::AlignVCenter);

    if (ZLineEdit* pLineEdit = qobject_cast<ZLineEdit*>(pControl)) {
        pLineEdit->setHintListWidget(m_hintlist.get(), m_descLabel.get());
    }
    else if (ZVecEditor* pVecEdit = qobject_cast<ZVecEditor*>(pControl)) {
        pVecEdit->setHintListWidget(m_hintlist.get(), m_descLabel.get());
    }
    else if (ZCodeEditor* pCodeEditor = qobject_cast<ZCodeEditor*>(pControl)) {
        pCodeEditor->setFuncDescLabel(m_descLabel.get());
    }

    _PANEL_CONTROL panelCtrl;
    panelCtrl.controlLayout = pLayout;
    panelCtrl.pLabel = pLabel;
    panelCtrl.pIconLabel = pIcon;
    panelCtrl.m_coreparamIdx = idxCoreParam;
    panelCtrl.pControl = pControl;

    if (pIcon)
        pIcon->setVisible(bVisible);
    if (pLabel)
        pLabel->setVisible(bVisible);
    if (pControl) {
        pControl->setVisible(bVisible);
        pControl->setEnabled(bEnable);
    }

    m_inputControls[tabName][groupName][paramName] = panelCtrl;

    if (bFloat && pControl) {
        m_floatControls << panelCtrl;
        pLabel->installEventFilter(this);
        pControl->installEventFilter(this);
        ZenoMainWindow* mainWin = zenoApp->getMainWindow();
        ZASSERT_EXIT(mainWin);
        ZTimeline* timeline = mainWin->timeline();
        ZASSERT_EXIT(timeline);
        onUpdateFrame(pControl, timeline->value(), idxCoreParam.data(ROLE_PARAM_VALUE));
        connect(timeline, &ZTimeline::sliderValueChanged, pControl, [=](int nFrame) {
            onUpdateFrame(pControl, nFrame, idxCoreParam.data(ROLE_PARAM_VALUE));
            }, Qt::UniqueConnection);
        connect(mainWin, &ZenoMainWindow::visFrameUpdated, pControl, [=](bool bGLView, int nFrame) {
            onUpdateFrame(pControl, nFrame, idxCoreParam.data(ROLE_PARAM_VALUE));
            }, Qt::UniqueConnection);
    }
}

void ZenoPropPanel::addOutputWidget(ZScrollArea* scrollArea, QGridLayout* pLayout, QStandardItem* pOutputItem, int row)
{
    ZASSERT_EXIT(pOutputItem);
    auto paramItem = pOutputItem->child(row);
    const QString& paramName = paramItem->data(ROLE_PARAM_NAME).toString();
    QPersistentModelIndex perIdx(paramItem->index());

    ZTextLabel* pLabel = new ZTextLabel(paramName);
    QFont font = QApplication::font();
    font.setWeight(QFont::Light);
    pLabel->setFont(font);
    pLabel->setToolTip(paramItem->data(ROLE_PARAM_TOOLTIP).toString());
    pLabel->setTextColor(QColor(255, 255, 255, 255 * 0.7));
    pLabel->setHoverCursor(Qt::ArrowCursor);
    //pLabel->setProperty("cssClass", "proppanel");

    bool bVisible = paramItem->data(ROLE_PARAM_SOCKET_VISIBLE).toBool();
    ZIconLabel* pIcon = new ZIconLabel();
    pIcon->setIcons(ZenoStyle::dpiScaledSize(QSize(26, 26)), ":/icons/parameter_key-frame_idle.svg", ":/icons/parameter_key-frame_hover.svg",
        ":/icons/parameter_key-frame_correct.svg", ":/icons/parameter_key-frame_correct.svg");
    pIcon->toggle(bVisible);
    connect(pIcon, &ZIconLabel::toggled, this, [=](bool toggled) {
        ParamsModel* paramsModel = QVariantPtr<ParamsModel>::asPtr(m_idx.data(ROLE_PARAMS));
        const QModelIndex& idx = paramsModel->paramIdx(perIdx.data(ROLE_PARAM_NAME).toString(), false);
        UiHelper::qIndexSetData(idx, toggled, ROLE_PARAM_SOCKET_VISIBLE);
        });

    pLayout->addWidget(pIcon, row, 0, Qt::AlignCenter);
    pLayout->addWidget(pLabel, row, 1, Qt::AlignLeft | Qt::AlignVCenter);

    _PANEL_CONTROL panelCtrl;
    panelCtrl.controlLayout = pLayout;
    panelCtrl.pLabel = pLabel;
    panelCtrl.pIconLabel = pIcon;
    panelCtrl.m_coreparamIdx = perIdx;
    panelCtrl.pControl = nullptr;

    m_outputControls[paramName] = panelCtrl;
}

void ZenoPropPanel::onViewParamAboutToBeRemoved(const QModelIndex& parent, int first, int last)
{
    if (m_inputControls.isEmpty() || !m_idx.isValid())
        return;

    QStandardItemModel* paramsModel = QVariantPtr<ParamsModel>::asPtr(m_idx.data(ROLE_PARAMS))->customParamModel();
    ZASSERT_EXIT(paramsModel);

    QStandardItem* parentItem = paramsModel->itemFromIndex(parent);
    if (parentItem == nullptr)
    {
        //clear all
        while (m_tabWidget->count() > 0)
        {
            QWidget* wid = m_tabWidget->widget(0);
            m_tabWidget->removeTab(0);
            delete wid;
        }
        return;
    }

    QStandardItem* removeItem = parentItem->child(first);
    if (removeItem->data(ROLE_PARAM_GROUP) == zeno::Role_InputPrimitive)
    {
        if (!m_idx.isValid())
            return;
        //subnet节点，可能有多个tab
        if ((m_idx.data(ROLE_NODETYPE) == zeno::Node_SubgraphNode || m_idx.data(ROLE_NODETYPE) == zeno::Node_AssetInstance)) {
            int vType = removeItem->data(ROLE_ELEMENT_TYPE).toInt();
            const QString& name = removeItem->data(ROLE_PARAM_NAME).toString();

            if (VPARAM_TAB == vType)
            {
                int idx = UiHelper::tabIndexOfName(m_tabWidget, name);
                m_tabWidget->removeTab(idx);
            }
            else if (VPARAM_GROUP == vType)
            {
                ZASSERT_EXIT(parentItem->data(ROLE_ELEMENT_TYPE) == VPARAM_TAB);
                const QString& tabName = parentItem->data(ROLE_PARAM_NAME).toString();
                int idx = UiHelper::tabIndexOfName(m_tabWidget, tabName);
                QWidget* tabWid = m_tabWidget->widget(idx);
                QVBoxLayout* pTabLayout = qobject_cast<QVBoxLayout*>(tabWid->layout());
                for (int i = 0; i < pTabLayout->count(); i++)
                {
                    QLayoutItem* pLayoutItem = pTabLayout->itemAt(i);
                    if (ZExpandableSection* pGroup = qobject_cast<ZExpandableSection*>(pLayoutItem->widget()))
                    {
                        if (pGroup->title() == name)
                        {
                            delete pGroup;
                            pTabLayout->removeItem(pLayoutItem);
                            break;
                        }
                    }
                }
            }
            else if (VPARAM_PARAM == vType)
            {
                ZASSERT_EXIT(parentItem->data(ROLE_ELEMENT_TYPE) == VPARAM_GROUP);

                QStandardItem* pTabItem = parentItem->parent();
                ZASSERT_EXIT(pTabItem && pTabItem->data(ROLE_ELEMENT_TYPE) == VPARAM_TAB);

                const QString& tabName = pTabItem->data(ROLE_PARAM_NAME).toString();
                const QString& groupName = parentItem->data(ROLE_PARAM_NAME).toString();
                const QString& paramName = name;

                ZExpandableSection* pGroupWidget = findGroup(tabName, groupName);
                if (!pGroupWidget)
                    return;

                QGridLayout* pGroupLayout = qobject_cast<QGridLayout*>(pGroupWidget->contentLayout());
                ZASSERT_EXIT(pGroupLayout);
                if (pGroupWidget->title() == groupName)
                {
                    _PANEL_CONTROL& ctrl = m_inputControls[tabName][groupName][paramName];
                    if (ctrl.controlLayout)
                    {
                        QGridLayout* pGridLayout = qobject_cast<QGridLayout*>(ctrl.controlLayout);
                        ZASSERT_EXIT(pGridLayout);

                        ctrl.controlLayout->removeWidget(ctrl.pControl);
                        delete ctrl.pControl;
                        if (ctrl.pLabel) {
                            ctrl.controlLayout->removeWidget(ctrl.pLabel);
                            delete ctrl.pLabel;
                        }
                        if (ctrl.pIconLabel) {
                            ctrl.controlLayout->removeWidget(ctrl.pIconLabel);
                            delete ctrl.pIconLabel;
                        }
                        m_inputControls[tabName][groupName].remove(paramName);
                    }
                }
            }
        }
        else {  //普通节点
            ZASSERT_EXIT(parentItem->data(ROLE_ELEMENT_TYPE) == VPARAM_GROUP);
            QStandardItem* pTabItem = parentItem->parent();
            ZASSERT_EXIT(pTabItem && pTabItem->data(ROLE_ELEMENT_TYPE) == VPARAM_TAB);

            if (ZScrollArea* area = qobject_cast<ZScrollArea*>(m_normalNodeInputWidget)) {
                QList<QGridLayout*> layouts = area->findChildren<QGridLayout*>();
                if (!layouts.empty() && layouts[0]) {
                    const auto& tabName = pTabItem->data(ROLE_PARAM_NAME).toString();
                    const auto& groupName = parentItem->data(ROLE_PARAM_NAME).toString();
                    const auto& paramName = removeItem->data(ROLE_PARAM_NAME).toString();
                    _PANEL_CONTROL& ctrl = m_inputControls[tabName][groupName][paramName];
                    if (ctrl.controlLayout)
                    {
                        QGridLayout* pGridLayout = qobject_cast<QGridLayout*>(ctrl.controlLayout);
                        ZASSERT_EXIT(pGridLayout);

                        ctrl.controlLayout->removeWidget(ctrl.pControl);
                        delete ctrl.pControl;
                        if (ctrl.pLabel) {
                            ctrl.controlLayout->removeWidget(ctrl.pLabel);
                            delete ctrl.pLabel;
                        }
                        if (ctrl.pIconLabel) {
                            ctrl.controlLayout->removeWidget(ctrl.pIconLabel);
                            delete ctrl.pIconLabel;
                        }
                        m_inputControls[tabName][groupName].remove(paramName);
                    }
                }
            }
        }
    } else if (removeItem->data(ROLE_PARAM_GROUP) == zeno::Role_OutputPrimitive) {
        if (ZScrollArea* area = qobject_cast<ZScrollArea*>(m_outputWidget)) {
            QList<QGridLayout*> layouts = area->findChildren<QGridLayout*>();
            if (!layouts.empty() && layouts[0]) {
                const auto& paramName = removeItem->data(ROLE_PARAM_NAME).toString();
                _PANEL_CONTROL& ctrl = m_outputControls[paramName];
                if (ctrl.controlLayout)
                {
                    QGridLayout* pGridLayout = qobject_cast<QGridLayout*>(ctrl.controlLayout);
                    ZASSERT_EXIT(pGridLayout);

                    ctrl.controlLayout->removeWidget(ctrl.pControl);
                    delete ctrl.pControl;
                    if (ctrl.pLabel) {
                        ctrl.controlLayout->removeWidget(ctrl.pLabel);
                        delete ctrl.pLabel;
                    }
                    if (ctrl.pIconLabel) {
                        ctrl.controlLayout->removeWidget(ctrl.pIconLabel);
                        delete ctrl.pIconLabel;
                    }
                    m_outputControls.remove(paramName);
                }
            }
        }
    }
}

void ZenoPropPanel::onCustomParamDataChanged(const QModelIndex& topLeft, const QModelIndex& bottomRight, const QVector<int>& roles)
{
    if (topLeft.data(ROLE_ELEMENT_TYPE) != VPARAM_PARAM || !m_idx.isValid() || (m_inputControls.isEmpty() && m_outputControls.isEmpty()))
        return;

    QStandardItemModel* paramsModel = QVariantPtr<ParamsModel>::asPtr(m_idx.data(ROLE_PARAMS))->customParamModel();
    ZASSERT_EXIT(paramsModel);

    QStandardItem* paramItem = paramsModel->itemFromIndex(topLeft);
    ZASSERT_EXIT(paramItem);

    if (topLeft.data(ROLE_PARAM_GROUP) == zeno::Role_InputPrimitive)
    {
        //现在一般都是改name或者value，暂时不区分subnet和普通节点
        QStandardItem* groupItem = paramItem->parent();
        ZASSERT_EXIT(groupItem);

        QStandardItem* tabItem = groupItem->parent();
        ZASSERT_EXIT(tabItem);

        int role = roles[0];
        const QString& tabName = tabItem->data(ROLE_PARAM_NAME).toString();
        const QString& groupName = groupItem->data(ROLE_PARAM_NAME).toString();

        PANEL_GROUP& group = m_inputControls[tabName][groupName];

        for (int r = topLeft.row(); r <= bottomRight.row(); r++)
        {
            QStandardItem* param = groupItem->child(r);

            if (role == ROLE_PARAM_NAME)
            {
                for (auto it = group.begin(); it != group.end(); it++)
                {
                    if (it->second.m_coreparamIdx == param->index())
                    {
                        const QString& newName = it->second.m_coreparamIdx.data(ROLE_PARAM_NAME).toString();
                        it->second.pLabel->setText(newName);
                        it->first = newName;
                        break;
                    }
                }
            }
            else if (role == ROLE_PARAM_CONTROL)
            {
                const QString& paramName = param->data(ROLE_PARAM_NAME).toString();
                _PANEL_CONTROL& ctrl = group[paramName];
                QGridLayout* pGridLayout = qobject_cast<QGridLayout*>(ctrl.controlLayout);
                ZASSERT_EXIT(pGridLayout);

                ctrl.controlLayout->removeWidget(ctrl.pControl);
                delete ctrl.pControl;
                if (ctrl.pLabel) {
                    ctrl.controlLayout->removeWidget(ctrl.pLabel);
                    delete ctrl.pLabel;
                }
                if (ctrl.pIconLabel) {
                    ctrl.controlLayout->removeWidget(ctrl.pIconLabel);
                    delete ctrl.pIconLabel;
                }

                int row = group.keys().indexOf(paramName, 0);
                ZExpandableSection* pExpand = findGroup(tabName, groupName);
                syncAddControl(pExpand, pGridLayout, param, row);
            }
            else if (role == ROLE_PARAM_VALUE)
            {
                const QString& paramName = param->data(ROLE_PARAM_NAME).toString();
                const QVariant& qvarAny = param->data(ROLE_PARAM_VALUE);
                zeno::reflect::Any value = qvarAny.value<zeno::reflect::Any>();
                ZASSERT_EXIT(value.has_value());

                _PANEL_CONTROL& ctrl = m_inputControls[tabName][groupName][paramName];
                BlockSignalScope scope(ctrl.pControl);

                if (QLineEdit* pLineEdit = qobject_cast<QLineEdit*>(ctrl.pControl))
                {
                    zeno::ParamType type = (zeno::ParamType)param->data(ROLE_PARAM_TYPE).toLongLong();
                    zeno::ParamControl paramCtrl = (zeno::ParamControl)param->data(ROLE_PARAM_CONTROL).toInt();
                    QString literalNum;
                    if (type == zeno::types::gParamType_Float) {
                        QVariant newVal = qvarAny;
                        if (curve_util::getCurveValue(newVal)) {
                            literalNum = QString::number(newVal.toFloat());
                        } else {
                            literalNum = UiHelper::anyToString(value);
                        }
                        pLineEdit->setText(literalNum);
                        QVector<QString> properties = curve_util::getKeyFrameProperty(qvarAny);
                        if (properties.empty())
                            return;
                        pLineEdit->setProperty(g_setKey, properties.first());
                        pLineEdit->style()->unpolish(pLineEdit);
                        pLineEdit->style()->polish(pLineEdit);
                        pLineEdit->update();
                    } else {
                        literalNum = UiHelper::anyToString(value);
                        pLineEdit->setText(literalNum);
                    }
                }
                else if (QComboBox* pCombobox = qobject_cast<QComboBox*>(ctrl.pControl))
                {
                    const std::string& text = any_cast<std::string>(value);
                    pCombobox->setCurrentText(QString::fromStdString(text));
                }
                else if (QTextEdit* pTextEidt = qobject_cast<QTextEdit*>(ctrl.pControl))
                {
                    const std::string& text = any_cast<std::string>(value);
                    pTextEidt->setText(QString::fromStdString(text));
                }
                else if (ZVecEditor* pVecEdit = qobject_cast<ZVecEditor*>(ctrl.pControl))
                {
                    ZASSERT_EXIT(value.type().hash_code() == gParamType_VecEdit);
                    QVariant newVal = qvarAny;
                    if (curve_util::getCurveValue(newVal)) {
                        pVecEdit->setVec(newVal.value<zeno::vecvar>(), pVecEdit->isFloat());
                        if (pVecEdit->isFloat()) {
                            QVector<QString> properties = curve_util::getKeyFrameProperty(qvarAny);
                            pVecEdit->updateProperties(properties);
                        }
                    }
                    else {
                        pVecEdit->setVec(any_cast<zeno::vecvar>(value), pVecEdit->isFloat());
                    }
                }
                else if (QCheckBox* pCheckbox = qobject_cast<QCheckBox*>(ctrl.pControl))
                {
                    bool bChecked = any_cast<bool>(value);
                    pCheckbox->setCheckState(bChecked ? Qt::Checked : Qt::Unchecked);
                }
                else if (QSlider* pSlider = qobject_cast<QSlider*>(ctrl.pControl))
                {
                    int intval = any_cast<int>(value);
                    pSlider->setValue(intval);
                }
                else if (QSpinBox* pSpinBox = qobject_cast<QSpinBox*>(ctrl.pControl))
                {
                    int intval = any_cast<int>(value);
                    pSpinBox->setValue(intval);
                }
                else if (QDoubleSpinBox* pSpinBox = qobject_cast<QDoubleSpinBox*>(ctrl.pControl))
                {
                    float fval = any_cast<float>(value);
                    pSpinBox->setValue(fval);
                }
                else if (ZSpinBoxSlider* pSpinSlider = qobject_cast<ZSpinBoxSlider*>(ctrl.pControl))
                {
                    int intval = any_cast<int>(value);
                    pSpinSlider->setValue(intval);
                }
                else if (QPushButton *pBtn = qobject_cast<QPushButton *>(ctrl.pControl))
                {
                    // colorvec3f
                    //if (value.canConvert<UI_VECTYPE>()) {
                    //    UI_VECTYPE vec = value.value<UI_VECTYPE>();
                    //    if (vec.size() == 3) {
                    //        auto color = QColor::fromRgbF(vec[0], vec[1], vec[2]);
                    //        pBtn->setStyleSheet(QString("background-color:%1; border:0;").arg(color.name()));
                    //    }
                    //}
                }
                //...
            }
            else if (role == ROLE_PARAM_CTRL_PROPERTIES)
            {
                const QString &paramName = param->data(ROLE_PARAM_NAME).toString();
                const QVariant &value = param->data(ROLE_PARAM_CTRL_PROPERTIES);
                    _PANEL_CONTROL &ctrl = m_inputControls[tabName][groupName][paramName];
                BlockSignalScope scope(ctrl.pControl);

                if (QComboBox *pCombobox = qobject_cast<QComboBox *>(ctrl.pControl))
                {
                    if (value.type() == QMetaType::QVariantMap && value.toMap().contains("items"))
                    {
                        pCombobox->clear();
                        pCombobox->addItems(value.toMap()["items"].toStringList());
                    }
                }
                else if (value.type() == QMetaType::QVariantMap &&
                    (value.toMap().contains("min") || value.toMap().contains("max") || value.toMap().contains("step")))
                {
                    QVariantMap map = value.toMap();
                    SLIDER_INFO info;
                    if (map.contains("min")) {
                        info.min = map["min"].toDouble();
                    }
                    if (map.contains("max")) {
                        info.max = map["max"].toDouble();
                    }
                    if (map.contains("step")) {
                        info.step = map["step"].toDouble();
                    }

                    if (qobject_cast<ZSpinBoxSlider *>(ctrl.pControl))
                    {
                        ZSpinBoxSlider *pSpinBoxSlider = qobject_cast<ZSpinBoxSlider *>(ctrl.pControl);
                        pSpinBoxSlider->setSingleStep(info.step);
                        pSpinBoxSlider->setRange(info.min, info.max);
                    }
                    else if (qobject_cast<QSlider *>(ctrl.pControl))
                    {
                        QSlider *pSlider = qobject_cast<QSlider *>(ctrl.pControl);
                        pSlider->setSingleStep(info.step);
                        pSlider->setRange(info.min, info.max);
                    }
                    else if (qobject_cast<QSpinBox *>(ctrl.pControl))
                    {
                        QSpinBox *pSpinBox = qobject_cast<QSpinBox *>(ctrl.pControl);
                        pSpinBox->setSingleStep(info.step);
                        pSpinBox->setRange(info.min, info.max);
                    }
                    else if (qobject_cast<QDoubleSpinBox *>(ctrl.pControl))
                    {
                        QDoubleSpinBox *pSpinBox = qobject_cast<QDoubleSpinBox *>(ctrl.pControl);
                        pSpinBox->setSingleStep(info.step);
                        pSpinBox->setRange(info.min, info.max);
                    }
                }
            }
            else if (role == ROLE_PARAM_TOOLTIP)
            {
                for (auto it = group.begin(); it != group.end(); it++)
                {
                    if (it->second.m_coreparamIdx == param->index())
                    {
                        const QString &newTip = it->second.m_coreparamIdx.data(ROLE_PARAM_TOOLTIP).toString();
                        it->second.pLabel->setToolTip(newTip);
                        break;
                    }
                }
            }
            else if (role == ROLE_PARAM_SOCKET_VISIBLE)
            {
                for (auto it = group.begin(); it != group.end(); it++)
                {
                    if (it->second.m_coreparamIdx == param->index())
                    {
                        bool socketVisible = it->second.m_coreparamIdx.data(ROLE_PARAM_SOCKET_VISIBLE).toBool();
                        if (ZIconLabel* icon = qobject_cast<ZIconLabel*>(it->second.pIconLabel)) {
                            icon->toggle(socketVisible);
                        }
                        break;
                    }
                }
            }
        }
    }
    else if (topLeft.data(ROLE_PARAM_GROUP) == zeno::Role_OutputPrimitive) {
        int role = roles[0];
        const QString& paramName = paramItem->data(ROLE_PARAM_NAME).toString();

        for (int r = topLeft.row(); r <= bottomRight.row(); r++) {
            if (role == ROLE_PARAM_NAME) {
                for (auto it = m_outputControls.begin(); it != m_outputControls.end(); it++) {
                    if (it->second.m_coreparamIdx == paramItem->index()) {
                        const QString& newName = it->second.m_coreparamIdx.data(ROLE_PARAM_NAME).toString();
                        it->second.pLabel->setText(newName);
                        it->first = newName;
                        break;
                    }
                }
            }
            else if (role == ROLE_PARAM_SOCKET_VISIBLE)
            {
                for (auto it = m_outputControls.begin(); it != m_outputControls.end(); it++)
                {
                    if (it->second.m_coreparamIdx == paramItem->index())
                    {
                        bool socketVisible = it->second.m_coreparamIdx.data(ROLE_PARAM_SOCKET_VISIBLE).toBool();
                        if (ZIconLabel* icon = qobject_cast<ZIconLabel*>(it->second.pIconLabel)) {
                            icon->toggle(socketVisible);
                        }
                        break;
                    }
                }
            }
        }
    }
}

void ZenoPropPanel::onViewParamsMoved(const QModelIndex &parent, int start, int end, const QModelIndex &destination, int destRow) 
{
    QStandardItemModel* viewParams = QVariantPtr<ParamsModel>::asPtr(m_idx.data(ROLE_PARAMS))->customParamModel();
    QStandardItem *parentItem = viewParams->itemFromIndex(parent);
    ZASSERT_EXIT(parentItem);
    ZASSERT_EXIT(parentItem->data(ROLE_VPARAM_TYPE) == VPARAM_GROUP);
    if (parent != destination || start == destRow)
        return;

    const QString &groupName = parentItem->text();
    QStandardItem *pTabItem = parentItem->parent();
    ZASSERT_EXIT(pTabItem);
    const QString &tabName = pTabItem->text();
    const QString &paramName = parentItem->child(start)->text();
    QGridLayout *pGridLayout = qobject_cast<QGridLayout *>(m_inputControls[tabName][groupName][paramName].controlLayout);
    ZASSERT_EXIT(pGridLayout);
    for (int row = 0; row < pGridLayout->rowCount(); row++) 
    {
        const QString &name = parentItem->child(row)->text();
        _PANEL_CONTROL control = m_inputControls[tabName][groupName][name];
        QWidget *labelWidget = nullptr;
        if (pGridLayout->itemAtPosition(row, 1))
            labelWidget = pGridLayout->itemAtPosition(row, 1)->widget();
        if (control.pLabel != labelWidget) 
        {
            pGridLayout->addWidget(control.pLabel, row, 1, Qt::AlignLeft | Qt::AlignVCenter);
        }
        QWidget *controlWidget = nullptr;
        if (pGridLayout->itemAtPosition(row, 2))
            controlWidget = pGridLayout->itemAtPosition(row, 2)->widget();
        if (control.pControl != controlWidget) 
        {
            pGridLayout->addWidget(control.pControl, row, 2, Qt::AlignVCenter);
        }
    }
}

void ZenoPropPanel::onLinkAdded(const zeno::EdgeInfo& link)
{
    if (m_dictListLinksTable) {
        m_dictListLinksTable->addLink(link);
    }
}

void ZenoPropPanel::onLinkRemoved(const zeno::EdgeInfo& link)
{
    if (m_dictListLinksTable) {
        m_dictListLinksTable->removeLink(link);
    }
}

void ZenoPropPanel::onDictListTableUpdateLink(QList<QPair<QString, QModelIndex>> links)
{
    if (GraphModel* currGraph = QVariantPtr<GraphModel>::asPtr(m_idx.data(ROLE_GRAPH))) {
        for (auto& [inkey, link] : links) {
            zeno::EdgeInfo edge = link.data(ROLE_LINK_INFO).value<zeno::EdgeInfo>();
            currGraph->updateLink(link, true, QString::fromStdString(edge.inKey), inkey);
        }
    }
}

void ZenoPropPanel::onDictListTableRemoveLink(QList<QModelIndex> links)
{
    if (GraphModel* currGraph = QVariantPtr<GraphModel>::asPtr(m_idx.data(ROLE_GRAPH))) {
        for (auto& link : links) {
            zeno::EdgeInfo edge = link.data(ROLE_LINK_INFO).value<zeno::EdgeInfo>();
            currGraph->removeLink(edge);
        }
    }
}

void ZenoPropPanel::onSettings()
{
    QMenu* pMenu = new QMenu(this);
    pMenu->setAttribute(Qt::WA_DeleteOnClose);

    QAction* pEditLayout = new QAction(tr("Edit Parameter Layout"));
    pMenu->addAction(pEditLayout);
    connect(pEditLayout, &QAction::triggered, [=]() {
        if (!m_idx.isValid())
            return;
        ParamsModel* paramsM = QVariantPtr<ParamsModel>::asPtr(m_idx.data(ROLE_PARAMS));

        QStandardItemModel* viewParams = QVariantPtr<ParamsModel>::asPtr(m_idx.data(ROLE_PARAMS))->customParamModel();
        ZASSERT_EXIT(viewParams);

        if (m_idx.data(ROLE_NODETYPE) != zeno::Node_SubgraphNode) 
        {
            QMessageBox::information(this, tr("Info"), tr("Cannot edit parameters!"));
            return;
        }
        ZEditParamLayoutDlg dlg(viewParams, this);
        if (QDialog::Accepted == dlg.exec())
        {
            zeno::ParamsUpdateInfo info = dlg.getEdittedUpdateInfo();
            paramsM->resetCustomUi(dlg.getCustomUiInfo());
            paramsM->batchModifyParams(info);
        }
    });
    pMenu->exec(QCursor::pos());
}

bool ZenoPropPanel::eventFilter(QObject *obj, QEvent *event) 
{
    if (event->type() == QEvent::ContextMenu) {
        for (auto ctrl : m_floatControls) {
            if (ctrl.pControl == obj || ctrl.pLabel == obj) {
                //get curves
                QStringList keys = curve_util::getKeys(obj, ctrl.m_coreparamIdx.data(ROLE_PARAM_VALUE), ctrl.pControl, ctrl.pLabel);
                zeno::CurvesData curves = curve_util::getCurvesData(ctrl.m_coreparamIdx, keys);
                //show menu
                QMenu *menu = new QMenu;
                QAction setAction(tr("Set KeyFrame"));
                QAction delAction(tr("Del KeyFrame"));
                QAction kFramesAction(tr("KeyFrames"));
                QAction clearAction(tr("Clear KeyFrames"));

                //set action enable
                int nSize = getKeyFrameSize(curves);
                delAction.setEnabled(nSize != 0);
                setAction.setEnabled(curves.keys.empty() || nSize != curves.keys.size());
                kFramesAction.setEnabled(!curves.keys.empty());
                clearAction.setEnabled(!curves.keys.empty());
                //add action
                menu->addAction(&setAction);
                menu->addAction(&delAction);
                menu->addAction(&kFramesAction);
                menu->addAction(&clearAction);
                //set key frame
                connect(&setAction, &QAction::triggered, this, [=]() { 
                    setKeyFrame(ctrl, keys); 
                });
                //del key frame
                connect(&delAction, &QAction::triggered, this, [=]() { 
                    delKeyFrame(ctrl, keys); 
                });
                //edit key frame
                connect(&kFramesAction, &QAction::triggered, this, [=]() {
                    editKeyFrame(ctrl, keys);
                });
                //clear key frame
                connect(&clearAction, &QAction::triggered, this, [=]() {
                    clearKeyFrame(ctrl, keys);
                });

                menu->exec(QCursor::pos());
                menu->deleteLater();
                return true;
            }
        }
    }
    return QWidget::eventFilter(obj, event);
}

void ZenoPropPanel::paintEvent(QPaintEvent* event)
{
    QStyleOption opt;
    opt.init(this);
    QPainter p(this);
    style()->drawPrimitive(QStyle::PE_Widget, &opt, &p, this);
}

void ZenoPropPanel::onNodeRemoved(QString nodeName)
{
    if (m_idx.row() < 0)
        clearLayout();
}


void ZenoPropPanel::setKeyFrame(const _PANEL_CONTROL &ctrl, const QStringList &keys) 
{
    if (ZCoreParamLineEdit* lineEdit = qobject_cast<ZCoreParamLineEdit*>(ctrl.pControl)) {
        lineEdit->setKeyFrame(keys);
    } else if (ZVecEditor* vecEdit = qobject_cast<ZVecEditor*>(ctrl.pControl)) {
        vecEdit->setKeyFrame(keys);
    }
    curve_util::updateTimelineKeys(m_idx.data(ROLE_KEYFRAMES).value<QVector<int>>());
}

void ZenoPropPanel::delKeyFrame(const _PANEL_CONTROL &ctrl, const QStringList &keys) 
{
    if (ZCoreParamLineEdit* lineEdit = qobject_cast<ZCoreParamLineEdit*>(ctrl.pControl)) {
        lineEdit->delKeyFrame(keys);
    } else if (ZVecEditor* vecEdit = qobject_cast<ZVecEditor*>(ctrl.pControl)) {
        vecEdit->delKeyFrame(keys);
    }
    curve_util::updateTimelineKeys(m_idx.data(ROLE_KEYFRAMES).value<QVector<int>>());
}

void ZenoPropPanel::editKeyFrame(const _PANEL_CONTROL &ctrl, const QStringList &keys) 
{
    if (ZCoreParamLineEdit* lineEdit = qobject_cast<ZCoreParamLineEdit*>(ctrl.pControl)) {
        lineEdit->editKeyFrame(keys);
    } else if (ZVecEditor* vecEdit = qobject_cast<ZVecEditor*>(ctrl.pControl)) {
        vecEdit->editKeyFrame(keys);
    }
    curve_util::updateTimelineKeys(m_idx.data(ROLE_KEYFRAMES).value<QVector<int>>());
}

void ZenoPropPanel::clearKeyFrame(const _PANEL_CONTROL& ctrl, const QStringList& keys)
{
    if (ZCoreParamLineEdit* lineEdit = qobject_cast<ZCoreParamLineEdit*>(ctrl.pControl)) {
        lineEdit->clearKeyFrame(keys);
    } else if (ZVecEditor* vecEdit = qobject_cast<ZVecEditor*>(ctrl.pControl)) {
        vecEdit->clearKeyFrame(keys);
    }
    curve_util::updateTimelineKeys(m_idx.data(ROLE_KEYFRAMES).value<QVector<int>>());
}

void ZenoPropPanel::onUpdateFrame(QWidget* pContrl, int nFrame, QVariant val)
{
    if (ParamsModel* paramsModel = QVariantPtr<ParamsModel>::asPtr(m_idx.data(ROLE_PARAMS))) {
        if (ZCoreParamLineEdit* lineEdit = qobject_cast<ZCoreParamLineEdit*>(pContrl)) {
            if (lineEdit->serKeyFrameStyle(val)) {
                paramsModel->setData(m_idx, true, ROLE_NODE_DIRTY);
            }
        }
        else if (ZVecEditor* vecEdit = qobject_cast<ZVecEditor*>(pContrl)) {
            if (vecEdit->serKeyFrameStyle(val)) {
                paramsModel->setData(m_idx, true, ROLE_NODE_DIRTY);
            }
        }
    }
}
