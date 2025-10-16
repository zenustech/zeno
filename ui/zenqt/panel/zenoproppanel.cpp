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
#include <zeno/utils/helper.h>
#include <zeno/core/typeinfo.h>
#include "declmetatype.h"
#include "model/customuimodel.h"


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
    , m_hintlist(new ZenoHintListWidget(this))
    , m_descLabel(new ZenoFuncDescriptionLabel(this))
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

    oldValue = m_idx.data(QtRole::ROLE_NODE_NAME).toString();
    if (value == oldValue)
        return true;

    if (GraphModel* pModel = QVariantPtr<GraphModel>::asPtr(m_idx.data(QtRole::ROLE_GRAPH)))
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

    if (m_idx.data(QtRole::ROLE_CLASS_NAME).toString() == "MakeDict" || m_idx.data(QtRole::ROLE_CLASS_NAME).toString() == "MakeList") {
        clearMakeDictMakeListLayout();
            }
    else {
        int nodeType = m_idx.data(QtRole::ROLE_NODETYPE).toInt();
        if (nodeType == zeno::Node_SubgraphNode || nodeType == zeno::Node_AssetInstance) {
            QString clsname = m_idx.data(QtRole::ROLE_CLASS_NAME).toString();
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
			ParamsModel* paramsM = QVariantPtr<ParamsModel>::asPtr(m_idx.data(QtRole::ROLE_PARAMS));
            if (paramsM) {
                disconnect(paramsM, &ParamsModel::dataChanged, this, &ZenoPropPanel::onParamModelDataChanged);
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

    if (m_idx.data(QtRole::ROLE_CLASS_NAME).toString() == "MakeDict" || m_idx.data(QtRole::ROLE_CLASS_NAME).toString() == "MakeList") {
        if (QWidget* wid = resetMakeDictMakeListLayout()) {
            pMainLayout->addWidget(wid);
            }
        }
    else {
        ParamsModel* paramsM = QVariantPtr<ParamsModel>::asPtr(m_idx.data(QtRole::ROLE_PARAMS));
        ZASSERT_EXIT(paramsM);
		connect(m_model, &GraphModel::nodeRemoved, this, &ZenoPropPanel::onNodeRemoved, Qt::UniqueConnection);
		connect(paramsM, &ParamsModel::enabledVisibleChanged, this, &ZenoPropPanel::onUpdateParamsVisbleEnabled);

		CustomUIModel* customUiM = paramsM->customUIModel();
        ZASSERT_EXIT(customUiM)

        const auto& initWidgets = [this, pMainLayout, customUiM] {
			int nodeType = m_idx.data(QtRole::ROLE_NODETYPE).toInt();
			if (nodeType == zeno::Node_SubgraphNode || nodeType == zeno::Node_AssetInstance) {
				//子图节点布局
				if (QWidget* wid = resetSubnetLayout()) {
					pMainLayout->addWidget(wid);
				}
			} else {
				ParamTabModel* tableM = customUiM->tabModel();
                ZASSERT_EXIT(tableM)
				ParamGroupModel* groupM = tableM->data(tableM->index(0), QmlCUIRole::GroupModel).value<ParamGroupModel*>();
                ZASSERT_EXIT(groupM);
				//普通节点布局，有多个group或tab也按subnet处理
				if (tableM->rowCount() != 1 || groupM->rowCount() != 1) {
					if (QWidget* wid = resetSubnetLayout())
						pMainLayout->addWidget(wid);
				}
				else {
					if (QWidget* wid = resetNormalNodeLayout())
						pMainLayout->addWidget(wid);
				}
			}
        };

		connect(paramsM, &ParamsModel::dataChanged, this, &ZenoPropPanel::onParamModelDataChanged);
        connect(customUiM, &CustomUIModel::resetCustomuiModel, [this, initWidgets, paramsM]() {
			clearLayout();
			connect(paramsM, &ParamsModel::dataChanged, this, &ZenoPropPanel::onParamModelDataChanged);
            initWidgets();
        });

        initWidgets();
    }
}

void ZenoPropPanel::onUpdateParamsVisbleEnabled()
{
    bool bUpdate = true;
    for (auto& [_, tab] : m_inputControls) {
        for (auto& [_, group] : tab) {
            for (auto& [_, control] : group) {
                bool bEnable = control.m_coreparamIdx.data(QtRole::ROLE_PARAM_ENABLE).toBool();
                bool bVisible = control.m_coreparamIdx.data(QtRole::ROLE_PARAM_VISIBLE).toBool();

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
    if (ParamsModel* paramsModel = QVariantPtr<ParamsModel>::asPtr(m_idx.data(QtRole::ROLE_PARAMS))) {
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
        if (ParamsModel* paramsModel = QVariantPtr<ParamsModel>::asPtr(m_idx.data(QtRole::ROLE_PARAMS))) {
            disconnect(paramsModel, &ParamsModel::linkAboutToBeRemoved, this, &ZenoPropPanel::onLinkRemoved);
            disconnect(paramsModel, &ParamsModel::linkAboutToBeInserted, this, &ZenoPropPanel::onLinkAdded);
            disconnect(m_dictListLinksTable, &ZenoDictListLinksTable::linksUpdated, this, &ZenoPropPanel::onDictListTableUpdateLink);
            disconnect(m_dictListLinksTable, &ZenoDictListLinksTable::linksRemoved, this, &ZenoPropPanel::onDictListTableRemoveLink);
        }
    }
}

#if 0
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
#endif

QWidget* ZenoPropPanel::resetSubnetLayout()
{
    CustomUIModel* customuiM = QVariantPtr<ParamsModel>::asPtr(m_idx.data(QtRole::ROLE_PARAMS))->customUIModel();
    if (!customuiM) {
        return nullptr;
    }
    ParamTabModel* tablmodel = customuiM->tabModel();
    if (!tablmodel) {
        return nullptr;
    }

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

    for (int i = 0; i < tablmodel->rowCount(); i++)
    {
		ParamGroupModel* pGroupM = tablmodel->data(tablmodel->index(i), QmlCUIRole::GroupModel).value<ParamGroupModel*>();
        if (!pGroupM) {
            continue;
        }
		QString tabname = tablmodel->data(tablmodel->index(i), QtRole::ROLE_PARAM_NAME).toString();
        syncAddTab(m_tabWidget, pGroupM, tabname, i);
    }

    m_descLabel->updateParent();
    m_hintlist->updateParent();
    m_hintlist->resetSize();

    splitter->addWidget(m_tabWidget);
    update();
    if (QWidget* outputsWidget = resetOutputs()) {
        splitter->addWidget(outputsWidget);
    }
    splitter->setStretchFactor(0, 3);
    splitter->setStretchFactor(1, 1);
    return splitter;
}

bool ZenoPropPanel::syncAddControl(ZExpandableSection* pGroupWidget, QGridLayout* pGroupLayout, ParamPlainModel* paramM, int row)
{
    ZASSERT_EXIT(paramM && pGroupLayout, false);
    ParamGroupModel* pGroupM = paramM->groupModel();
    ZASSERT_EXIT(pGroupM, false);
    ParamTabModel* pTabM = pGroupM->tabModel();
    ZASSERT_EXIT(pTabM, false);

    const QString& tabName = pGroupM->parentName();
    const QString& groupName = paramM->parentName();

    const QString& paramName = paramM->data(paramM->index(row), QtRole::ROLE_PARAM_NAME).toString();
    const zeno::reflect::Any& anyVal = paramM->data(paramM->index(row), QtRole::ROLE_PARAM_VALUE).value<zeno::reflect::Any>();
    //QVariant val = UiHelper::anyToQvar();
    zeno::ParamControl ctrl = (zeno::ParamControl)paramM->data(paramM->index(row), QtRole::ROLE_PARAM_CONTROL).toInt();

    const zeno::ParamType type = (zeno::ParamType)paramM->data(paramM->index(row), QtRole::ROLE_PARAM_TYPE).toLongLong();
    const zeno::reflect::Any &pros = paramM->data(paramM->index(row), QtRole::ROLE_PARAM_CTRL_PROPERTIES).value<zeno::reflect::Any>();

    QPersistentModelIndex perIdx(paramM->getParamsModelIndex(row));
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
            //if (!AppHelper::updateCurve(paramItem->data(QtRole::ROLE_PARAM_VALUE), newValue))
            //{
                //onCustomParamDataChanged(perIdx, perIdx, QVector<int>() << QtRole::ROLE_PARAM_VALUE);
                //return;
            //}
            QStandardItemModel* paramsModel = QVariantPtr<ParamsModel>::asPtr(m_idx.data(QtRole::ROLE_PARAMS))->customParamModel();
            BlockSignalScope scope(paramsModel); //setData时需屏蔽dataChange信号
            paramsModel->setData(perIdx, newValue, QtRole::ROLE_PARAM_VALUE);
        }
        */
        ParamsModel* paramsModel = QVariantPtr<ParamsModel>::asPtr(m_idx.data(QtRole::ROLE_PARAMS));
        const QModelIndex& idx = paramsModel->paramIdx(perIdx.data(QtRole::ROLE_PARAM_NAME).toString(), true);
        UiHelper::qIndexSetData(idx, QVariant::fromValue(newValue), QtRole::ROLE_PARAM_VALUE);
    };
    cbSet.cbSwitch = [=](bool bOn) {
        zenoApp->getMainWindow()->setInDlgEventLoop(bOn);   //deal with ubuntu dialog slow problem when update viewport.
    };
    cbSet.cbGetIndexData = [=]() -> QVariant { 
        return perIdx.isValid() ? paramM->data(paramM->index(row), QtRole::ROLE_PARAM_VALUE) : QVariant();
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
    pLabel->setToolTip(paramM->data(paramM->index(row), QtRole::ROLE_PARAM_TOOLTIP).toString());

    pLabel->setTextColor(QColor(255, 255, 255, 255 * 0.7));
    pLabel->setHoverCursor(Qt::ArrowCursor);
    //pLabel->setProperty("cssClass", "proppanel");
    bool bVisible = paramM->data(paramM->index(row), QtRole::ROLE_PARAM_SOCKET_VISIBLE).toBool();

    ZIconLabel* pIcon = new ZIconLabel();
    pIcon->setIcons(ZenoStyle::dpiScaledSize(QSize(26, 26)), ":/icons/parameter_key-frame_idle.svg", ":/icons/parameter_key-frame_hover.svg",
        ":/icons/parameter_key-frame_correct.svg", ":/icons/parameter_key-frame_correct.svg");
    pIcon->toggle(bVisible);
    connect(pIcon, &ZIconLabel::toggled, this, [=](bool toggled) {
        ParamsModel* paramsModel = QVariantPtr<ParamsModel>::asPtr(m_idx.data(QtRole::ROLE_PARAMS));
        const QModelIndex& idx = paramsModel->paramIdx(perIdx.data(QtRole::ROLE_PARAM_NAME).toString(), true);
        UiHelper::qIndexSetData(idx, toggled, QtRole::ROLE_PARAM_SOCKET_VISIBLE);
    });

    ZIconLabel* poriginIcon = nullptr, *pnewIcon = pIcon;
    ZTextLabel* poriginLabel = nullptr, *pnewLabel = pLabel;
    QWidget* poriginControl = nullptr, *pnewControl = pControl;
    if (row < pGroupLayout->rowCount()) {//考虑pGroupLayout中插入行的情况
        while (pGroupLayout->itemAtPosition(row, 0) && pGroupLayout->itemAtPosition(row, 1) && pControl && pGroupLayout->itemAtPosition(row, 2)) {
            poriginIcon = qobject_cast<ZIconLabel*>(pGroupLayout->itemAtPosition(row, 0)->widget());
            pGroupLayout->replaceWidget(poriginIcon, pnewIcon);
            pnewIcon = poriginIcon;

            poriginLabel = qobject_cast<ZTextLabel*>(pGroupLayout->itemAtPosition(row, 1)->widget());
            pGroupLayout->replaceWidget(poriginLabel, pnewLabel);
            pnewLabel = poriginLabel;

            poriginControl = pGroupLayout->itemAtPosition(row, 2)->widget();
            pGroupLayout->replaceWidget(poriginControl, pnewControl);
            pnewControl = poriginControl;
            row++;
        }
    }
    pGroupLayout->addWidget(pnewIcon, row, 0, Qt::AlignCenter);

    pGroupLayout->addWidget(pnewLabel, row, 1, Qt::AlignLeft | Qt::AlignVCenter);
    if (pControl)
        pGroupLayout->addWidget(pnewControl, row, 2, Qt::AlignVCenter);

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
        onUpdateFrame(pControl, timeline->value(), paramM->data(paramM->index(row), QtRole::ROLE_PARAM_VALUE));
        connect(timeline, &ZTimeline::sliderValueChanged, pControl, [=](int nFrame) {
            onUpdateFrame(pControl, nFrame, paramM->data(paramM->index(row), QtRole::ROLE_PARAM_VALUE));
            }, Qt::UniqueConnection);
        connect(mainWin, &ZenoMainWindow::visFrameUpdated, pControl, [=](bool bGLView, int nFrame) {
            onUpdateFrame(pControl, nFrame, paramM->data(paramM->index(row), QtRole::ROLE_PARAM_VALUE));
            }, Qt::UniqueConnection);
    }
    return true;
}

bool ZenoPropPanel::syncAddGroup(QVBoxLayout* pTabLayout, ParamPlainModel* paramM, QString groupName, int row)
{
    bool bCollaspe = false;//TOOD: collasped.
    ZExpandableSection* pGroupWidget = new ZExpandableSection(groupName);
    pGroupWidget->setObjectName(groupName);
    pGroupWidget->setCollasped(bCollaspe);
    QGridLayout* pLayout = new QGridLayout;
    pLayout->setContentsMargins(10, 15, 10, 15);
    //pLayout->setColumnStretch(1, 1);
    pLayout->setColumnStretch(2, 3);
    pLayout->setSpacing(10);
    for (int k = 0; k < paramM->rowCount(); k++)
    {
        syncAddControl(pGroupWidget, pLayout, paramM, k);
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

bool ZenoPropPanel::syncAddTab(QTabWidget* pTabWidget, ParamGroupModel* groupM, QString tabName, int row)
{
    QWidget* tabWid = pTabWidget->widget(UiHelper::tabIndexOfName(pTabWidget, tabName));
    if (tabWid)
        return false;

    QWidget* pTabWid = new QWidget;
    QVBoxLayout* pTabLayout = new QVBoxLayout;
    pTabLayout->setContentsMargins(QMargins(0, 0, 0, 0));
    pTabLayout->setSpacing(0);
    if (m_idx.data(QtRole::ROLE_NODETYPE) == zeno::Node_Group) 
    {
        ZenoBlackboardPropWidget *propWidget = new ZenoBlackboardPropWidget(m_idx, pTabWid);
        pTabLayout->addWidget(propWidget);
    } 
    else 
    {
        for (int j = 0; j < groupM->rowCount(); j++) {
            ParamPlainModel* paramM = groupM->data(groupM->index(j), QmlCUIRole::PrimModel).value<ParamPlainModel*>();
            if (!paramM) {
                continue;
            }
            QString groupname = groupM->data(groupM->index(j), QtRole::ROLE_PARAM_NAME).toString();
            syncAddGroup(pTabLayout, paramM, groupname, j);
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
	CustomUIModel* customuiM = QVariantPtr<ParamsModel>::asPtr(m_idx.data(QtRole::ROLE_PARAMS))->customUIModel();
    ZASSERT_EXIT(customuiM, nullptr);
	ParamTabModel* tablmodel = customuiM->tabModel();
    ZASSERT_EXIT(tablmodel, nullptr);
	ParamGroupModel* pGroupM = tablmodel->data(tablmodel->index(0), QmlCUIRole::GroupModel).value<ParamGroupModel*>();
    ZASSERT_EXIT(pGroupM, nullptr);
	ParamPlainModel* paramM = pGroupM->data(pGroupM->index(0), QmlCUIRole::PrimModel).value<ParamPlainModel*>();
    ZASSERT_EXIT(paramM, nullptr);

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

    for (int row = 0; row < paramM->rowCount(); row++) {
        normalNodeAddInputWidget(scrollArea, pLayout, paramM, row);
    }

    m_descLabel->updateParent();
    m_hintlist->updateParent();
    m_hintlist->resetSize();

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
	CustomUIModel* customuiM = QVariantPtr<ParamsModel>::asPtr(m_idx.data(QtRole::ROLE_PARAMS))->customUIModel();
    ZASSERT_EXIT(customuiM, nullptr);
    PrimParamOutputModel* primOutputModel = customuiM->primOutputModel();
    ZASSERT_EXIT(primOutputModel, nullptr);

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

    for (int row = 0; row < primOutputModel->rowCount(); row++) {
        addOutputWidget(scrollArea, pLayout, primOutputModel, row);
    }
    m_outputWidget = scrollArea;
    return m_outputWidget;
}

void ZenoPropPanel::normalNodeAddInputWidget(ZScrollArea* scrollArea, QGridLayout* pLayout, ParamPlainModel* paramM, int row)
{
	ZASSERT_EXIT(paramM);
	ParamGroupModel* pGroupM = paramM->groupModel();
	ZASSERT_EXIT(pGroupM);
	ParamTabModel* pTabM = pGroupM->tabModel();
	ZASSERT_EXIT(pTabM);

    const QString& tabName = pGroupM->parentName();
	const QString& groupName = paramM->parentName();
	const QString& paramName = paramM->data(paramM->index(row), QtRole::ROLE_PARAM_NAME).toString();

    zeno::reflect::Any anyVal = paramM->data(paramM->index(row), QtRole::ROLE_PARAM_VALUE).value<zeno::reflect::Any>();
    const zeno::ParamType type = (zeno::ParamType)paramM->data(paramM->index(row), QtRole::ROLE_PARAM_TYPE).toLongLong();

    bool bInput = paramM->data(paramM->index(row), QtRole::ROLE_ISINPUT).toBool();
    ParamsModel* paramsM = QVariantPtr<ParamsModel>::asPtr(m_idx.data(QtRole::ROLE_PARAMS));
    QPersistentModelIndex idxCoreParam = paramsM->paramIdx(paramName, bInput);

    zeno::ParamControl ctrl = (zeno::ParamControl)idxCoreParam.data(QtRole::ROLE_PARAM_CONTROL).toInt();

    const zeno::reflect::Any& pros = idxCoreParam.data(QtRole::ROLE_PARAM_CTRL_PROPERTIES).value<zeno::reflect::Any>();

    CallbackCollection cbSet;

    bool bFloat = UiHelper::isFloatType(type);
    cbSet.cbEditFinished = [=](zeno::reflect::Any newValue) {
        paramsM->setData(idxCoreParam, QVariant::fromValue(newValue), QtRole::ROLE_PARAM_VALUE);
        //paramItem->setData(QVariant::fromValue(newValue), QtRole::ROLE_PARAM_VALUE);
    };
    cbSet.cbSwitch = [=](bool bOn) {
        zenoApp->getMainWindow()->setInDlgEventLoop(bOn);   //deal with ubuntu dialog slow problem when update viewport.
    };
    cbSet.cbGetIndexData = [=]() -> QVariant {
        return idxCoreParam.isValid() ? idxCoreParam.data(QtRole::ROLE_PARAM_VALUE) : QVariant();
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
    pLabel->setToolTip(idxCoreParam.data(QtRole::ROLE_PARAM_TOOLTIP).toString());

    pLabel->setTextColor(QColor(255, 255, 255, 255 * 0.7));
    pLabel->setHoverCursor(Qt::ArrowCursor);
    //pLabel->setProperty("cssClass", "proppanel");

    bool bVisible = idxCoreParam.data(QtRole::ROLE_PARAM_VISIBLE).toBool();
    bool bEnable = idxCoreParam.data(QtRole::ROLE_PARAM_ENABLE).toBool();
    bool bSocketVisible = idxCoreParam.data(QtRole::ROLE_PARAM_SOCKET_VISIBLE).toBool();
    ZIconLabel* pIcon = new ZIconLabel();
    pIcon->setIcons(ZenoStyle::dpiScaledSize(QSize(26, 26)), ":/icons/parameter_key-frame_idle.svg", ":/icons/parameter_key-frame_hover.svg",
        ":/icons/parameter_key-frame_correct.svg", ":/icons/parameter_key-frame_correct.svg");
    pIcon->toggle(bSocketVisible);
    connect(pIcon, &ZIconLabel::toggled, this, [=](bool toggled) {
        UiHelper::qIndexSetData(idxCoreParam, toggled, QtRole::ROLE_PARAM_SOCKET_VISIBLE);
        });
    pLayout->addWidget(pIcon, row, 0, Qt::AlignCenter);

    pLayout->addWidget(pLabel, row, 1, Qt::AlignLeft | Qt::AlignVCenter);
    if (pControl) {
		if (ZCodeEditor* pCodeEditor = qobject_cast<ZCodeEditor*>(pControl)) {
			pCodeEditor->setFixedHeight(this->height() - ZenoStyle::dpiScaled(60));
		}
        pLayout->addWidget(pControl, row, 2, Qt::AlignVCenter);
    }

    if (ZLineEdit* pLineEdit = qobject_cast<ZLineEdit*>(pControl)) {
        pLineEdit->setHintListWidget(m_hintlist.get(), m_descLabel.get());
    }
    else if (ZVecEditor* pVecEdit = qobject_cast<ZVecEditor*>(pControl)) {
        pVecEdit->setHintListWidget(m_hintlist.get(), m_descLabel.get());
    }
    else if (ZCodeEditor* pCodeEditor = qobject_cast<ZCodeEditor*>(pControl)) {
        pCodeEditor->setHintListWidget(m_hintlist.get(), m_descLabel.get());
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
        onUpdateFrame(pControl, timeline->value(), idxCoreParam.data(QtRole::ROLE_PARAM_VALUE));
        connect(timeline, &ZTimeline::sliderValueChanged, pControl, [=](int nFrame) {
            onUpdateFrame(pControl, nFrame, idxCoreParam.data(QtRole::ROLE_PARAM_VALUE));
            }, Qt::UniqueConnection);
        connect(mainWin, &ZenoMainWindow::visFrameUpdated, pControl, [=](bool bGLView, int nFrame) {
            onUpdateFrame(pControl, nFrame, idxCoreParam.data(QtRole::ROLE_PARAM_VALUE));
            }, Qt::UniqueConnection);
    }
}

void ZenoPropPanel::addOutputWidget(ZScrollArea* scrollArea, QGridLayout* pLayout, PrimParamOutputModel* primOutputM, int row)
{
    ZASSERT_EXIT(primOutputM);
    const QString& paramName = primOutputM->data(primOutputM->index(row), QtRole::ROLE_PARAM_NAME).toString();
    QPersistentModelIndex perIdx(primOutputM->getParamsModelIndex(row));

    ZTextLabel* pLabel = new ZTextLabel(paramName);
    QFont font = QApplication::font();
    font.setWeight(QFont::Light);
    pLabel->setFont(font);
    pLabel->setToolTip(primOutputM->data(primOutputM->index(row), QtRole::ROLE_PARAM_TOOLTIP).toString());
    pLabel->setTextColor(QColor(255, 255, 255, 255 * 0.7));
    pLabel->setHoverCursor(Qt::ArrowCursor);
    //pLabel->setProperty("cssClass", "proppanel");

    bool bVisible = primOutputM->data(primOutputM->index(row), QtRole::ROLE_PARAM_SOCKET_VISIBLE).toBool();
    ZIconLabel* pIcon = new ZIconLabel();
    pIcon->setIcons(ZenoStyle::dpiScaledSize(QSize(26, 26)), ":/icons/parameter_key-frame_idle.svg", ":/icons/parameter_key-frame_hover.svg",
        ":/icons/parameter_key-frame_correct.svg", ":/icons/parameter_key-frame_correct.svg");
    pIcon->toggle(bVisible);
    connect(pIcon, &ZIconLabel::toggled, this, [=](bool toggled) {
        ParamsModel* paramsModel = QVariantPtr<ParamsModel>::asPtr(m_idx.data(QtRole::ROLE_PARAMS));
        const QModelIndex& idx = paramsModel->paramIdx(perIdx.data(QtRole::ROLE_PARAM_NAME).toString(), false);
        UiHelper::qIndexSetData(idx, toggled, QtRole::ROLE_PARAM_SOCKET_VISIBLE);
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

void ZenoPropPanel::onParamModelDataChanged(const QModelIndex& topLeft, const QModelIndex& bottomRight, const QVector<int>& roles)
{
    ParamsModel* paramsM = QVariantPtr<ParamsModel>::asPtr(m_idx.data(QtRole::ROLE_PARAMS));
    ZASSERT_EXIT(paramsM);
    CustomUIModel* customuiM = paramsM->customUIModel();
    ZASSERT_EXIT(customuiM);

	if (!m_idx.isValid() || (m_inputControls.isEmpty() && m_outputControls.isEmpty()))
		return;

	if (topLeft.data(QtRole::ROLE_PARAM_GROUP) == zeno::Role_InputPrimitive)
	{
        QString tabName, groupName, ctrlName;
        const auto& getCtrl = [this, &tabName, &groupName, &ctrlName](QModelIndex idx) {
            for (auto& [tn, tab] : m_inputControls) {
                for (auto& [gn, group] : tab) {
                    for (auto& [cn, ctrl] : group) {
                        if (ctrl.m_coreparamIdx == idx) {
                            tabName = tn;
                            groupName = gn;
                            ctrlName = cn;
                            return;
                        }
                    }
                }
            }
        };
		int role = roles[0];

        for (int r = topLeft.row(); r <= bottomRight.row(); r++) {
            QModelIndex idx = paramsM->index(r);
            getCtrl(idx);

            ParamTabModel* tabM = customuiM->tabModel();
            if (!tabM) {
                continue;
            }
            ParamGroupModel* groupM = tabM->data(tabM->indexFromName(tabName), QmlCUIRole::GroupModel).value<ParamGroupModel*>();
            if (!groupM) {
                continue;
            }
            ParamPlainModel* plainM = groupM->data(groupM->indexFromName(groupName), QmlCUIRole::PrimModel).value<ParamPlainModel*>();
            if (!plainM) {
                continue;
            }

			PANEL_GROUP& group = m_inputControls[tabName][groupName];

			if (role == QtRole::ROLE_PARAM_NAME)
			{

				for (auto it = group.begin(); it != group.end(); it++)
				{
					if (it->second.m_coreparamIdx == idx)
					{
						const QString& newName = it->second.m_coreparamIdx.data(QtRole::ROLE_PARAM_NAME).toString();
						it->second.pLabel->setText(newName);
						it->first = newName;
						break;
					}
				}
			}
			else if (role == QtRole::ROLE_PARAM_CONTROL)
			{
				const QString& paramName = idx.data(QtRole::ROLE_PARAM_NAME).toString();
				const zeno::ParamType type = (zeno::ParamType)idx.data(QtRole::ROLE_PARAM_TYPE).toLongLong();
				zeno::reflect::Any defAnyval = zeno::initAnyDeflValue(type);
				zeno::convertToEditVar(defAnyval, type);
				zeno::scope_exit sp([&paramsM] {paramsM->blockSignals(false); });
                paramsM->blockSignals(true);
                paramsM->setData(idx, QVariant::fromValue(defAnyval), QtRole::ROLE_PARAM_VALUE);

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

				syncAddControl(pExpand, pGridLayout, plainM, row);
			}
			else if (role == QtRole::ROLE_PARAM_VALUE)
			{
				const QString& paramName = idx.data(QtRole::ROLE_PARAM_NAME).toString();;
				const QVariant& qvarAny = idx.data(QtRole::ROLE_PARAM_VALUE);
				zeno::reflect::Any value = qvarAny.value<zeno::reflect::Any>();
				ZASSERT_EXIT(value.has_value());

				_PANEL_CONTROL& ctrl = m_inputControls[tabName][groupName][paramName];
				BlockSignalScope scope(ctrl.pControl);

				if (QLineEdit* pLineEdit = qobject_cast<QLineEdit*>(ctrl.pControl))
				{
					zeno::ParamType type = (zeno::ParamType)idx.data(QtRole::ROLE_PARAM_TYPE).toLongLong();
					zeno::ParamControl paramCtrl = (zeno::ParamControl)idx.data(QtRole::ROLE_PARAM_CONTROL).toInt();
					QString literalNum;
					if (type == zeno::types::gParamType_Float) {
						QVariant newVal = qvarAny;
						if (curve_util::getCurveValue(newVal)) {
							literalNum = QString::number(newVal.toFloat());
						}
						else {
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
					}
					else {
						literalNum = UiHelper::anyToString(value);
						pLineEdit->setText(literalNum);
					}
				}
				else if (QComboBox* pCombobox = qobject_cast<QComboBox*>(ctrl.pControl))
				{
					const std::string& text = zeno::any_cast_to_string(value);
					pCombobox->setCurrentText(QString::fromStdString(text));
				}
				else if (QTextEdit* pTextEidt = qobject_cast<QTextEdit*>(ctrl.pControl))
				{
					const std::string& text = zeno::any_cast_to_string(value);
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
				else if (QPushButton* pBtn = qobject_cast<QPushButton*>(ctrl.pControl))
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
			}
			else if (role == QtRole::ROLE_PARAM_CTRL_PROPERTIES)
			{
				const QString& paramName = idx.data(QtRole::ROLE_PARAM_NAME).toString();
				const QVariant& value = idx.data(QtRole::ROLE_PARAM_CTRL_PROPERTIES);
				_PANEL_CONTROL& ctrl = m_inputControls[tabName][groupName][paramName];
				BlockSignalScope scope(ctrl.pControl);

				if (QComboBox* pCombobox = qobject_cast<QComboBox*>(ctrl.pControl))
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

					if (qobject_cast<ZSpinBoxSlider*>(ctrl.pControl))
					{
						ZSpinBoxSlider* pSpinBoxSlider = qobject_cast<ZSpinBoxSlider*>(ctrl.pControl);
						pSpinBoxSlider->setSingleStep(info.step);
						pSpinBoxSlider->setRange(info.min, info.max);
					}
					else if (qobject_cast<QSlider*>(ctrl.pControl))
					{
						QSlider* pSlider = qobject_cast<QSlider*>(ctrl.pControl);
						pSlider->setSingleStep(info.step);
						pSlider->setRange(info.min, info.max);
					}
					else if (qobject_cast<QSpinBox*>(ctrl.pControl))
					{
						QSpinBox* pSpinBox = qobject_cast<QSpinBox*>(ctrl.pControl);
						pSpinBox->setSingleStep(info.step);
						pSpinBox->setRange(info.min, info.max);
					}
					else if (qobject_cast<QDoubleSpinBox*>(ctrl.pControl))
					{
						QDoubleSpinBox* pSpinBox = qobject_cast<QDoubleSpinBox*>(ctrl.pControl);
						pSpinBox->setSingleStep(info.step);
						pSpinBox->setRange(info.min, info.max);
					}
				}
				}
			else if (role == QtRole::ROLE_PARAM_TOOLTIP)
			{
				for (auto it = group.begin(); it != group.end(); it++)
				{
					if (it->second.m_coreparamIdx == idx)
					{
						const QString& newTip = it->second.m_coreparamIdx.data(QtRole::ROLE_PARAM_TOOLTIP).toString();
						it->second.pLabel->setToolTip(newTip);
						break;
					}
				}
				}
			else if (role == QtRole::ROLE_PARAM_SOCKET_VISIBLE)
			{
				for (auto it = group.begin(); it != group.end(); it++)
				{
					if (it->second.m_coreparamIdx == idx)
					{
						bool socketVisible = it->second.m_coreparamIdx.data(QtRole::ROLE_PARAM_SOCKET_VISIBLE).toBool();
						if (ZIconLabel* icon = qobject_cast<ZIconLabel*>(it->second.pIconLabel)) {
							icon->toggle(socketVisible);
						}
						break;
					}
				}
			}
        }
	}
	else if (topLeft.data(QtRole::ROLE_PARAM_GROUP) == zeno::Role_OutputPrimitive) {
		int role = roles[0];

		for (int r = topLeft.row(); r <= bottomRight.row(); r++) {
			QModelIndex idx = paramsM->index(r);

			if (role == QtRole::ROLE_PARAM_NAME) {
				for (auto it = m_outputControls.begin(); it != m_outputControls.end(); it++) {
					if (it->second.m_coreparamIdx == idx) {
						const QString& newName = it->second.m_coreparamIdx.data(QtRole::ROLE_PARAM_NAME).toString();
						it->second.pLabel->setText(newName);
						it->first = newName;
						break;
					}
				}
			}
			else if (role == QtRole::ROLE_PARAM_SOCKET_VISIBLE)
			{
				for (auto it = m_outputControls.begin(); it != m_outputControls.end(); it++)
				{
					if (it->second.m_coreparamIdx == idx)
					{
						bool socketVisible = it->second.m_coreparamIdx.data(QtRole::ROLE_PARAM_SOCKET_VISIBLE).toBool();
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
    if (GraphModel* currGraph = QVariantPtr<GraphModel>::asPtr(m_idx.data(QtRole::ROLE_GRAPH))) {
        for (auto& [inkey, link] : links) {
            zeno::EdgeInfo edge = link.data(QtRole::ROLE_LINK_INFO).value<zeno::EdgeInfo>();
            currGraph->updateLink(link, true, QString::fromStdString(edge.inKey), inkey);
        }
    }
}

void ZenoPropPanel::onDictListTableRemoveLink(QList<QModelIndex> links)
{
    if (GraphModel* currGraph = QVariantPtr<GraphModel>::asPtr(m_idx.data(QtRole::ROLE_GRAPH))) {
        for (auto& link : links) {
            zeno::EdgeInfo edge = link.data(QtRole::ROLE_LINK_INFO).value<zeno::EdgeInfo>();
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
        ParamsModel* paramsM = QVariantPtr<ParamsModel>::asPtr(m_idx.data(QtRole::ROLE_PARAMS));

        CustomUIModel* viewParams = QVariantPtr<ParamsModel>::asPtr(m_idx.data(QtRole::ROLE_PARAMS))->customUIModel();
        ZASSERT_EXIT(viewParams);

        if (m_idx.data(QtRole::ROLE_NODETYPE) != zeno::Node_SubgraphNode) 
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
                QStringList keys = curve_util::getKeys(obj, ctrl.m_coreparamIdx.data(QtRole::ROLE_PARAM_VALUE), ctrl.pControl, ctrl.pLabel);
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

void ZenoPropPanel::resizeEvent(QResizeEvent* e)
{
    for (auto& [_, tab] : m_inputControls) {
        for (auto& [_, group] : tab) {
            for (auto& [_, control] : group) {
                if (ZCodeEditor* pCodeEditor = qobject_cast<ZCodeEditor*>(control.pControl)) {
                    pCodeEditor->setFixedHeight(this->height() - ZenoStyle::dpiScaled(60));
                }
            }
        }
    }
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
    curve_util::updateTimelineKeys(m_idx.data(QtRole::ROLE_KEYFRAMES).value<QVector<int>>());
}

void ZenoPropPanel::delKeyFrame(const _PANEL_CONTROL &ctrl, const QStringList &keys) 
{
    if (ZCoreParamLineEdit* lineEdit = qobject_cast<ZCoreParamLineEdit*>(ctrl.pControl)) {
        lineEdit->delKeyFrame(keys);
    } else if (ZVecEditor* vecEdit = qobject_cast<ZVecEditor*>(ctrl.pControl)) {
        vecEdit->delKeyFrame(keys);
    }
    curve_util::updateTimelineKeys(m_idx.data(QtRole::ROLE_KEYFRAMES).value<QVector<int>>());
}

void ZenoPropPanel::editKeyFrame(const _PANEL_CONTROL &ctrl, const QStringList &keys) 
{
    if (ZCoreParamLineEdit* lineEdit = qobject_cast<ZCoreParamLineEdit*>(ctrl.pControl)) {
        lineEdit->editKeyFrame(keys);
    } else if (ZVecEditor* vecEdit = qobject_cast<ZVecEditor*>(ctrl.pControl)) {
        vecEdit->editKeyFrame(keys);
    }
    curve_util::updateTimelineKeys(m_idx.data(QtRole::ROLE_KEYFRAMES).value<QVector<int>>());
}

void ZenoPropPanel::clearKeyFrame(const _PANEL_CONTROL& ctrl, const QStringList& keys)
{
    if (ZCoreParamLineEdit* lineEdit = qobject_cast<ZCoreParamLineEdit*>(ctrl.pControl)) {
        lineEdit->clearKeyFrame(keys);
    } else if (ZVecEditor* vecEdit = qobject_cast<ZVecEditor*>(ctrl.pControl)) {
        vecEdit->clearKeyFrame(keys);
    }
    curve_util::updateTimelineKeys(m_idx.data(QtRole::ROLE_KEYFRAMES).value<QVector<int>>());
}

void ZenoPropPanel::onUpdateFrame(QWidget* pContrl, int nFrame, QVariant val)
{
    if (ParamsModel* paramsModel = QVariantPtr<ParamsModel>::asPtr(m_idx.data(QtRole::ROLE_PARAMS))) {
        if (ZCoreParamLineEdit* lineEdit = qobject_cast<ZCoreParamLineEdit*>(pContrl)) {
            if (lineEdit->serKeyFrameStyle(val)) {
                paramsModel->setData(m_idx, true, QtRole::ROLE_NODE_DIRTY);
            }
        }
        else if (ZVecEditor* vecEdit = qobject_cast<ZVecEditor*>(pContrl)) {
            if (vecEdit->serKeyFrameStyle(val)) {
                paramsModel->setData(m_idx, true, QtRole::ROLE_NODE_DIRTY);
            }
        }
    }
}
