#include "zenoproppanel.h"
#include "zenoapplication.h"
#include <zenomodel/include/graphsmanagment.h>
#include <zenomodel/include/modelrole.h>
#include <zenomodel/include/igraphsmodel.h>
#include <zenomodel/include/curvemodel.h>
#include "variantptr.h"
#include <zenoui/comctrl/zcombobox.h>
#include <zenoui/comctrl/zlabel.h>
#include <zenoui/style/zenostyle.h>
#include <zenoui/comctrl/gv/zenoparamwidget.h>
#include <zenoui/comctrl/zveceditor.h>
#include <zenomodel/include/uihelper.h>
#include <zenoui/comctrl/zexpandablesection.h>
#include <zenoui/comctrl/zlinewidget.h>
#include <zenoui/comctrl/zlineedit.h>
#include <zenoui/comctrl/ztextedit.h>
#include <zenoui/comctrl/zwidgetfactory.h>
#include "util/log.h"
#include "util/apphelper.h"
#include <zenomodel/include/curveutil.h>
#include <zenoui/comctrl/dialog/curvemap/zqwtcurvemapeditor.h>
#include <zenoui/comctrl/dialog/zenoheatmapeditor.h>
#include "zenomainwindow.h"
#include <zenomodel/include/viewparammodel.h>
#include "../dialog/zeditparamlayoutdlg.h"
#include <zenoui/comctrl/zspinboxslider.h>
#include "zenoblackboardpropwidget.h"
#include "timeline/ztimeline.h"
#include "util/apphelper.h"


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
{
    QVBoxLayout* pVLayout = new QVBoxLayout;
    pVLayout->setContentsMargins(QMargins(0, 0, 0, 0));
    setLayout(pVLayout);
    setFocusPolicy(Qt::ClickFocus);

    QPalette palette = this->palette();
    palette.setBrush(QPalette::Window, QColor("#2d3239"));
    setPalette(palette);
    setAutoFillBackground(true);
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

    oldValue = m_idx.data(ROLE_CUSTOM_OBJNAME).toString();
    if (value == oldValue)
        return true;

    bool isValid = false;
    IGraphsModel *pGraphsModel = zenoApp->graphsManagment()->currentModel();
    if (pGraphsModel) {
        isValid = pGraphsModel->setCustomName(m_subgIdx, m_idx, value);
        if (!isValid) {
            QMessageBox::warning(nullptr, tr("Warring"), tr("CustomName invalid!"));
        }
    }
    return isValid;
}

void ZenoPropPanel::clearLayout()
{
    setUpdatesEnabled(false);
    qDeleteAll(findChildren<QWidget*>(QString(), Qt::FindDirectChildrenOnly));
    QVBoxLayout* pMainLayout = qobject_cast<QVBoxLayout*>(this->layout());
    while (pMainLayout->count() > 0)
    {
        QLayoutItem* pItem = pMainLayout->itemAt(pMainLayout->count() - 1);
        pMainLayout->removeItem(pItem);
    }
    setUpdatesEnabled(true);
    m_tabWidget = nullptr;
    m_controls.clear();
    m_floatColtrols.clear();

    if (m_idx.isValid())
    {
        QStandardItemModel* paramsModel = QVariantPtr<QStandardItemModel>::asPtr(m_idx.data(ROLE_PANEL_PARAMS));
        if (paramsModel)
        {
            disconnect(paramsModel, &QStandardItemModel::rowsInserted, this, &ZenoPropPanel::onViewParamInserted);
            disconnect(paramsModel, &QStandardItemModel::rowsAboutToBeRemoved, this, &ZenoPropPanel::onViewParamAboutToBeRemoved);
            disconnect(paramsModel, &QStandardItemModel::dataChanged, this, &ZenoPropPanel::onViewParamDataChanged);
            disconnect(paramsModel, &QStandardItemModel::rowsMoved, this, &ZenoPropPanel::onViewParamsMoved);
        }
    }

    update();
}

void ZenoPropPanel::reset(IGraphsModel* pModel, const QModelIndex& subgIdx, const QModelIndexList& nodes, bool select)
{
    if (m_bReentry)
        return;

    RetryScope scope(m_bReentry);

    clearLayout();
    QVBoxLayout *pMainLayout = qobject_cast<QVBoxLayout *>(this->layout());

    if (!pModel || !select || nodes.size() != 1)
    {
        update();
        return;
    }

    m_subgIdx = subgIdx;
    m_idx = nodes[0];
    if (!m_idx.isValid())
        return;

    QStandardItemModel* paramsModel = QVariantPtr<QStandardItemModel>::asPtr(m_idx.data(ROLE_PANEL_PARAMS));
    if (!paramsModel)
        return;

    connect(paramsModel, &QStandardItemModel::rowsInserted, this, &ZenoPropPanel::onViewParamInserted);
    connect(paramsModel, &QStandardItemModel::rowsAboutToBeRemoved, this, &ZenoPropPanel::onViewParamAboutToBeRemoved);
    connect(paramsModel, &QStandardItemModel::dataChanged, this, &ZenoPropPanel::onViewParamDataChanged);
    connect(paramsModel, &QStandardItemModel::rowsMoved, this, &ZenoPropPanel::onViewParamsMoved);
    connect(paramsModel, &QStandardItemModel::modelAboutToBeReset, this, [=]() {
        //clear all
        if (m_tabWidget)
        {
            while (m_tabWidget->count() > 0)
            {
                QWidget *wid = m_tabWidget->widget(0);
                m_tabWidget->removeTab(0);
                delete wid;
            }
        }
    });
    connect(pModel, &IGraphsModel::_rowsRemoved, this, [=]() {
        clearLayout();
    });
    connect(pModel, &IGraphsModel::modelClear, this, [=]() {
        clearLayout();
    });

    QStandardItem* root = paramsModel->invisibleRootItem();
    if (!root) return;

    QStandardItem* pRoot = root->child(0);
    if (!pRoot) return;

    m_tabWidget = new QTabWidget;
    m_tabWidget->tabBar()->setProperty("cssClass", "propanel");
    m_tabWidget->setDocumentMode(true);
    m_tabWidget->setTabsClosable(false);
    m_tabWidget->setMovable(false);

    QFont font = QApplication::font();
    font.setWeight(QFont::Medium);

    m_tabWidget->setFont(font); //bug in qss font setting.
    m_tabWidget->tabBar()->setDrawBase(false);

    for (int i = 0; i < pRoot->rowCount(); i++)
    {
        QStandardItem* pTabItem = pRoot->child(i);
        syncAddTab(m_tabWidget, pTabItem, i);
    }

    pMainLayout->addWidget(m_tabWidget);
    pMainLayout->setSpacing(0);

    update();
}

void ZenoPropPanel::onViewParamInserted(const QModelIndex& parent, int first, int last)
{
    ZASSERT_EXIT(m_tabWidget);
    if (!m_idx.isValid())
        return;

    QStandardItemModel* paramsModel = QVariantPtr<QStandardItemModel>::asPtr(m_idx.data(ROLE_PANEL_PARAMS));
    ZASSERT_EXIT(paramsModel);

    if (!parent.isValid())
    {
        QStandardItem* root = paramsModel->invisibleRootItem();
        ZASSERT_EXIT(root);
        QStandardItem* pRoot = root->child(0);
        if (!pRoot) return;

        for (int i = 0; i < pRoot->rowCount(); i++)
        {
            QStandardItem* pTabItem = pRoot->child(i);
            syncAddTab(m_tabWidget, pTabItem, i);
        }
        return;
    }

    if (m_controls.isEmpty())
        return;

    QStandardItem* parentItem = paramsModel->itemFromIndex(parent);
    ZASSERT_EXIT(parentItem);
    QStandardItem* newItem = parentItem->child(first);
    int vType = newItem->data(ROLE_VPARAM_TYPE).toInt();
    const QString& name = newItem->data(ROLE_VPARAM_NAME).toString();
    if (vType == VPARAM_TAB)
    {
        syncAddTab(m_tabWidget, newItem, first);
    }
    else if (vType == VPARAM_GROUP)
    {
        ZASSERT_EXIT(parentItem->data(ROLE_VPARAM_TYPE) == VPARAM_TAB);
        const QString& tabName = parentItem->data(ROLE_VPARAM_NAME).toString();
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
        ZASSERT_EXIT(parentItem->data(ROLE_VPARAM_TYPE) == VPARAM_GROUP);

        QStandardItem* pTabItem = parentItem->parent();
        ZASSERT_EXIT(pTabItem && pTabItem->data(ROLE_VPARAM_TYPE) == VPARAM_TAB);

        const QString& tabName = pTabItem->data(ROLE_VPARAM_NAME).toString();
        const QString& groupName = parentItem->data(ROLE_VPARAM_NAME).toString();
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
    ViewParamModel *pModel = qobject_cast<ViewParamModel *>(sender());
    if (pModel && !newItem->data(ROLE_VPARAM_IS_COREPARAM).toBool())
        pModel->markDirty();
}

bool ZenoPropPanel::syncAddControl(ZExpandableSection* pGroupWidget, QGridLayout* pGroupLayout, QStandardItem* paramItem, int row)
{
    ZASSERT_EXIT(paramItem && pGroupLayout, false);
    QStandardItem* pGroupItem = paramItem->parent();
    ZASSERT_EXIT(pGroupItem, false);
    QStandardItem* pTabItem = pGroupItem->parent();
    ZASSERT_EXIT(pTabItem, false);

    const QString& tabName = pTabItem->data(ROLE_VPARAM_NAME).toString();
    const QString& groupName = pGroupItem->data(ROLE_VPARAM_NAME).toString();
    const QString& paramName = paramItem->data(ROLE_VPARAM_NAME).toString();
    QVariant val = paramItem->data(ROLE_PARAM_VALUE);
    PARAM_CONTROL ctrl = (PARAM_CONTROL)paramItem->data(ROLE_PARAM_CTRL).toInt();

    const QString &typeDesc = paramItem->data(ROLE_PARAM_TYPE).toString();
    const QVariant &pros = paramItem->data(ROLE_VPARAM_CTRL_PROPERTIES);

    QPersistentModelIndex perIdx(paramItem->index());
    CallbackCollection cbSet;

    if (ctrl == CONTROL_DICTPANEL)
    {
        val = paramItem->data(ROLE_VPARAM_LINK_MODEL);
        cbSet.cbNodeSelected = [=](const QModelIndex& outNodeIdx) {
            QAction act("Select Node");
            act.setProperty("ActionType", ZenoMainWindow::ACTION_SELECT_NODE);
            act.setData(outNodeIdx);

            RetryScope scope(m_bReentry);
            zenoApp->getMainWindow()->dispatchCommand(&act, true);
        };
    } 
    else if (ctrl == CONTROL_GROUP_LINE) 
    {
        return false;
    }

    bool bFloat = ctrl == CONTROL_VEC2_FLOAT || ctrl == CONTROL_VEC3_FLOAT || ctrl == CONTROL_VEC4_FLOAT || ctrl == CONTROL_FLOAT;
    cbSet.cbEditFinished = [=](QVariant newValue) {
        if (bFloat)
        {
            if (!AppHelper::updateCurve(paramItem->data(ROLE_PARAM_VALUE), newValue))
            {
                onViewParamDataChanged(perIdx, perIdx, QVector<int>() << ROLE_PARAM_VALUE);
                return;
            }
        }
        AppHelper::socketEditFinished(newValue, m_idx, perIdx);
        AppHelper::modifyOptixObjDirectly(newValue, m_idx, perIdx, true);
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
        bKeyFrame = AppHelper::getCurveValue(val);
    }
    QWidget* pControl = zenoui::createWidget(val, ctrl, typeDesc, cbSet, pros);

    ZTextLabel* pLabel = new ZTextLabel(paramName);

    QFont font = QApplication::font();
    font.setWeight(QFont::DemiBold);
    pLabel->setFont(font);
    pLabel->setToolTip(paramItem->data(ROLE_VPARAM_TOOLTIP).toString());

    pLabel->setTextColor(QColor(255, 255, 255, 255 * 0.7));
    pLabel->setHoverCursor(Qt::ArrowCursor);
    //pLabel->setProperty("cssClass", "proppanel");

    ZIconLabel *pIcon = new ZIconLabel;
    pIcon->setIcons(ZenoStyle::dpiScaledSize(QSize(24, 24)), ":/icons/parameter_key-frame_idle.svg", ":/icons/parameter_key-frame_hover.svg");
    pGroupLayout->addWidget(pIcon, row, 0, Qt::AlignCenter);

    pGroupLayout->addWidget(pLabel, row, 1, Qt::AlignLeft | Qt::AlignVCenter);
    if (pControl)
        pGroupLayout->addWidget(pControl, row, 2, Qt::AlignVCenter);

    if (ZTextEdit* pMultilineStr = qobject_cast<ZTextEdit*>(pControl))
    {
        connect(pMultilineStr, &ZTextEdit::geometryUpdated, pGroupWidget, &ZExpandableSection::updateGeo);
    }

    _PANEL_CONTROL panelCtrl;
    panelCtrl.controlLayout = pGroupLayout;
    panelCtrl.pLabel = pLabel;
    panelCtrl.pIcon = pIcon;
    panelCtrl.m_viewIdx = perIdx;
    panelCtrl.pControl = pControl;

    m_controls[tabName][groupName][paramName] = panelCtrl;

    if (bFloat) {
        m_floatColtrols << panelCtrl;
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
    bool bCollaspe = pGroupItem->data(ROLE_VPARAM_COLLASPED).toBool();
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

    QWidget* pTabWid = new QWidget;
    QVBoxLayout* pTabLayout = new QVBoxLayout;
    pTabLayout->setContentsMargins(QMargins(0, 0, 0, 0));
    pTabLayout->setSpacing(0);
    if (m_idx.data(ROLE_NODETYPE) == GROUP_NODE) 
    {
        ZenoBlackboardPropWidget *propWidget = new ZenoBlackboardPropWidget(m_idx, m_subgIdx, pTabWid);
        pTabLayout->addWidget(propWidget);
    } 
    else 
    {
        for (int j = 0; j < pTabItem->rowCount(); j++) {
            QStandardItem *pGroupItem = pTabItem->child(j);
            syncAddGroup(pTabLayout, pGroupItem, j);
        }
    }

    pTabLayout->addStretch();
    pTabWid->setLayout(pTabLayout);
    pTabWidget->insertTab(row, pTabWid, tabName);
    return true;
}

void ZenoPropPanel::onViewParamAboutToBeRemoved(const QModelIndex& parent, int first, int last)
{
    if (m_controls.isEmpty() || !m_idx.isValid())
        return;

    QStandardItemModel* paramsModel = QVariantPtr<QStandardItemModel>::asPtr(m_idx.data(ROLE_PANEL_PARAMS));
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
    int vType = removeItem->data(ROLE_VPARAM_TYPE).toInt();
    const QString& name = removeItem->data(ROLE_VPARAM_NAME).toString();

    if (VPARAM_TAB == vType)
    {
        int idx = UiHelper::tabIndexOfName(m_tabWidget, name);
        m_tabWidget->removeTab(idx);
    }
    else if (VPARAM_GROUP == vType)
    {
        ZASSERT_EXIT(parentItem->data(ROLE_VPARAM_TYPE) == VPARAM_TAB);
        const QString& tabName = parentItem->data(ROLE_VPARAM_NAME).toString();
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
        ZASSERT_EXIT(parentItem->data(ROLE_VPARAM_TYPE) == VPARAM_GROUP);

        QStandardItem* pTabItem = parentItem->parent();
        ZASSERT_EXIT(pTabItem && pTabItem->data(ROLE_VPARAM_TYPE) == VPARAM_TAB);

        const QString& tabName = pTabItem->data(ROLE_VPARAM_NAME).toString();
        const QString& groupName = parentItem->data(ROLE_VPARAM_NAME).toString();
        const QString& paramName = name;

        ZExpandableSection* pGroupWidget = findGroup(tabName, groupName);
        if (!pGroupWidget)  return;

        QGridLayout* pGroupLayout = qobject_cast<QGridLayout*>(pGroupWidget->contentLayout());
        ZASSERT_EXIT(pGroupLayout);
        if (pGroupWidget->title() == groupName)
        {
            _PANEL_CONTROL& ctrl = m_controls[tabName][groupName][paramName];
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
                if (ctrl.pIcon) {
                    ctrl.controlLayout->removeWidget(ctrl.pIcon);
                    delete ctrl.pIcon;
                }
                m_controls[tabName][groupName].remove(paramName);
            }
        }
    }
}

void ZenoPropPanel::onViewParamDataChanged(const QModelIndex& topLeft, const QModelIndex& bottomRight, const QVector<int>& roles)
{
    if (topLeft.data(ROLE_VPARAM_TYPE) != VPARAM_PARAM || !m_idx.isValid() || m_controls.isEmpty())
        return;

    QStandardItemModel* paramsModel = QVariantPtr<QStandardItemModel>::asPtr(m_idx.data(ROLE_PANEL_PARAMS));
    ZASSERT_EXIT(paramsModel);

    QStandardItem* paramItem = paramsModel->itemFromIndex(topLeft);
    ZASSERT_EXIT(paramItem);

    QStandardItem* groupItem = paramItem->parent();
    ZASSERT_EXIT(groupItem);

    QStandardItem* tabItem = groupItem->parent();
    ZASSERT_EXIT(tabItem);

    int role = roles[0];
    const QString& tabName = tabItem->data(ROLE_VPARAM_NAME).toString();
    const QString& groupName = groupItem->data(ROLE_VPARAM_NAME).toString();

    PANEL_GROUP& group = m_controls[tabName][groupName];

    for (int r = topLeft.row(); r <= bottomRight.row(); r++)
    {
        QStandardItem* param = groupItem->child(r);

        if (role == ROLE_PARAM_NAME)
        {
            for (auto it = group.begin(); it != group.end(); it++)
            {
                if (it->second.m_viewIdx == param->index())
                {
                    const QString& newName = it->second.m_viewIdx.data(ROLE_VPARAM_NAME).toString();
                    it->second.pLabel->setText(newName);
                    it->first = newName;
                    break;
                }
            }
        }
        else if (role == ROLE_PARAM_CTRL)
        {
            const QString& paramName = param->data(ROLE_VPARAM_NAME).toString();
            _PANEL_CONTROL& ctrl = group[paramName];
            QGridLayout* pGridLayout = qobject_cast<QGridLayout*>(ctrl.controlLayout);
            ZASSERT_EXIT(pGridLayout);

            ctrl.controlLayout->removeWidget(ctrl.pControl);
            delete ctrl.pControl;
            if (ctrl.pLabel) {
                ctrl.controlLayout->removeWidget(ctrl.pLabel);
                delete ctrl.pLabel;
            }
            if (ctrl.pIcon) {
                ctrl.controlLayout->removeWidget(ctrl.pIcon);
                delete ctrl.pIcon;
            }

            int row = group.keys().indexOf(paramName, 0);
            ZExpandableSection* pExpand = findGroup(tabName, groupName);
            syncAddControl(pExpand, pGridLayout, param, row);
        }
        else if (role == ROLE_PARAM_VALUE)
        {
            const QString& paramName = param->data(ROLE_VPARAM_NAME).toString();
            const QVariant& value = param->data(ROLE_PARAM_VALUE);
            _PANEL_CONTROL& ctrl = m_controls[tabName][groupName][paramName];
            BlockSignalScope scope(ctrl.pControl);

            if (QLineEdit* pLineEdit = qobject_cast<QLineEdit*>(ctrl.pControl))
            {
                PARAM_CONTROL paramCtrl = (PARAM_CONTROL)param->data(ROLE_PARAM_CTRL).toInt();
                QString literalNum;
                if (paramCtrl == CONTROL_FLOAT) {
                    QVariant newVal = value;
                    bool bKeyFrame = AppHelper::getCurveValue(newVal);
                    literalNum = UiHelper::variantToString(newVal);
                    pLineEdit->setText(literalNum);
                    QVector<QString> properties = AppHelper::getKeyFrameProperty(value);
                    pLineEdit->setProperty(g_setKey, properties.first());
                    pLineEdit->style()->unpolish(pLineEdit);
                    pLineEdit->style()->polish(pLineEdit);
                    pLineEdit->update();
                    
                } else {
                    literalNum = UiHelper::variantToString(value);
                    pLineEdit->setText(literalNum);
                }
            }
            else if (QComboBox* pCombobox = qobject_cast<QComboBox*>(ctrl.pControl))
            {
                pCombobox->setCurrentText(value.toString());
            }
            else if (QTextEdit* pTextEidt = qobject_cast<QTextEdit*>(ctrl.pControl))
            {
                pTextEidt->setText(value.toString());
            }
            else if (ZVecEditor* pVecEdit = qobject_cast<ZVecEditor*>(ctrl.pControl))
            {
                QVariant newVal = value;
                bool bKeyFrame = AppHelper::getCurveValue(newVal);
                pVecEdit->setVec(newVal, pVecEdit->isFloat());
                if (pVecEdit->isFloat())
                {
                    QVector<QString> properties = AppHelper::getKeyFrameProperty(value);
                    pVecEdit->updateProperties(properties);
                }
            }
            else if (QCheckBox* pCheckbox = qobject_cast<QCheckBox*>(ctrl.pControl))
            {
                pCheckbox->setCheckState(value.toBool() ? Qt::Checked : Qt::Unchecked);
            }
            else if (QSlider* pSlider = qobject_cast<QSlider*>(ctrl.pControl))
            {
                pSlider->setValue(value.toInt());
            }
            else if (QSpinBox* pSpinBox = qobject_cast<QSpinBox*>(ctrl.pControl))
            {
                pSpinBox->setValue(value.toInt());
            }
            else if (QDoubleSpinBox* pSpinBox = qobject_cast<QDoubleSpinBox*>(ctrl.pControl))
            {
                pSpinBox->setValue(value.toDouble());
            }
            else if (ZSpinBoxSlider* pSpinSlider = qobject_cast<ZSpinBoxSlider*>(ctrl.pControl))
            {
                pSpinSlider->setValue(value.toInt());
            } 
            else if (QPushButton *pBtn = qobject_cast<QPushButton *>(ctrl.pControl)) 
            {
                // colorvec3f
                if (value.canConvert<UI_VECTYPE>()) {
                    UI_VECTYPE vec = value.value<UI_VECTYPE>();
                    if (vec.size() == 3) {
                        auto color = QColor::fromRgbF(vec[0], vec[1], vec[2]);
                        pBtn->setStyleSheet(QString("background-color:%1; border:0;").arg(color.name()));
                    }
                }
            }
            //...
        }
		else if (role == ROLE_VPARAM_CTRL_PROPERTIES)
		{
            const QString &paramName = param->data(ROLE_VPARAM_NAME).toString();
            const QVariant &value = param->data(ROLE_VPARAM_CTRL_PROPERTIES);
            _PANEL_CONTROL &ctrl = m_controls[tabName][groupName][paramName];
            BlockSignalScope scope(ctrl.pControl);

            if (QComboBox *pCombobox = qobject_cast<QComboBox *>(ctrl.pControl)) 
			{
                if (value.type() == QMetaType::QVariantMap && value.toMap().contains("items"))
				{
                    pCombobox->clear();
                    pCombobox->addItems(value.toMap()["items"].toStringList());
				}
            } else if (value.type() == QMetaType::QVariantMap && 
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
        else if (role == ROLE_VPARAM_TOOLTIP) 
        {
            for (auto it = group.begin(); it != group.end(); it++) 
            {
                if (it->second.m_viewIdx == param->index()) 
                {
                    const QString &newTip = it->second.m_viewIdx.data(ROLE_VPARAM_TOOLTIP).toString();
                    it->second.pLabel->setToolTip(newTip);
                    break;
                }
            }
        }
    }
}

void ZenoPropPanel::onViewParamsMoved(const QModelIndex &parent, int start, int end, const QModelIndex &destination, int destRow) 
{
    QStandardItemModel *viewParams = QVariantPtr<QStandardItemModel>::asPtr(m_idx.data(ROLE_PANEL_PARAMS));
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
    QGridLayout *pGridLayout = qobject_cast<QGridLayout *>(m_controls[tabName][groupName][paramName].controlLayout);
    ZASSERT_EXIT(pGridLayout);
    for (int row = 0; row < pGridLayout->rowCount(); row++) 
    {
        const QString &name = parentItem->child(row)->text();
        _PANEL_CONTROL control = m_controls[tabName][groupName][name];
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

void ZenoPropPanel::getDelfCurveData(CURVE_DATA &curve, float y, bool visible, const QString &key) {
    curve.visible = visible;
    CURVE_RANGE &rg = curve.rg;
    rg.yFrom = rg.yFrom > y ? y : rg.yFrom;
    rg.yTo = rg.yTo > y ? rg.yTo : y;
    ZenoMainWindow *mainWin = zenoApp->getMainWindow();
    ZASSERT_EXIT(mainWin);
    ZTimeline *timeline = mainWin->timeline();
    ZASSERT_EXIT(timeline);
    QPair<int, int> fromTo = timeline->fromTo();
    rg.xFrom = fromTo.first;
    rg.xTo = fromTo.second;
    if (curve.points.isEmpty()) {
        curve.key = key;
        curve.cycleType = 0;
    }
    float x = timeline->value();
    CURVE_POINT point = {QPointF(x, y), QPointF(0, 0), QPointF(0, 0), HDL_ALIGNED};
    if (!curve.points.contains(point))
        curve.points.append(point);
    updateHandler(curve);
}

void ZenoPropPanel::updateHandler(CURVE_DATA &curve) {
    if (curve.points.size() > 1) {
        qSort(curve.points.begin(), curve.points.end(),
              [](const CURVE_POINT &p1, const CURVE_POINT &p2) { return p1.point.x() < p2.point.x(); });
        float preX = curve.points.at(0).point.x();
        for (int i = 1; i < curve.points.size(); i++) {
            QPointF p1 = curve.points.at(i - 1).point;
            QPointF p2 = curve.points.at(i).point;
            float distance = fabs(p1.x() - p2.x());
            float handle = distance * 0.2;
            if (i == 1) {
                curve.points[i - 1].leftHandler = QPointF(-handle, 0);
                curve.points[i - 1].rightHandler = QPointF(handle, 0);
            }
            if (p2.y() < p1.y() && (curve.points[i - 1].rightHandler.x() < 0)) {
                handle = -handle;
            }
            curve.points[i].leftHandler = QPointF(-handle, 0);
            curve.points[i].rightHandler = QPointF(handle, 0);
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

        QStandardItemModel* viewParams = QVariantPtr<QStandardItemModel>::asPtr(m_idx.data(ROLE_PANEL_PARAMS));
        ZASSERT_EXIT(viewParams);

        IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();
        if (!pGraphsModel->IsSubGraphNode(m_idx)) 
        {
            QMessageBox::information(this, tr("Info"), tr("Cannot edit parameters!"));
            return;
        }
        ZEditParamLayoutDlg dlg(viewParams, false, m_idx, pGraphsModel, this);
        dlg.exec();
    });
    pMenu->exec(QCursor::pos());
}

bool ZenoPropPanel::eventFilter(QObject *obj, QEvent *event) 
{
    if (event->type() == QEvent::ContextMenu) {
        for (auto ctrl : m_floatColtrols) {
            if (ctrl.pControl == obj || ctrl.pLabel == obj) {
                //get curves
                QStringList keys = getKeys(obj, ctrl);
                CURVES_DATA curves = getCurvesData(ctrl.m_viewIdx, keys);
                //show menu
                QMenu *menu = new QMenu;
                QAction setAction(tr("Set KeyFrame"));
                QAction delAction(tr("Del KeyFrame"));
                QAction kFramesAction(tr("KeyFrames"));
                QAction clearAction(tr("Clear KeyFrames"));

                //set action enable
                int nSize = getKeyFrameSize(curves);
                delAction.setEnabled(nSize != 0);
                setAction.setEnabled(curves.isEmpty() || nSize != curves.size());
                kFramesAction.setEnabled(!curves.isEmpty());
                clearAction.setEnabled(!curves.isEmpty());
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

void ZenoPropPanel::setKeyFrame(const _PANEL_CONTROL &ctrl, const QStringList &keys) 
{
    CURVES_DATA newVal;
    if (ctrl.m_viewIdx.data(ROLE_PARAM_VALUE).canConvert<CURVES_DATA>())
        newVal = ctrl.m_viewIdx.data(ROLE_PARAM_VALUE).value<CURVES_DATA>();
    UI_VECTYPE vec;
    if (ZLineEdit *lineEdit = qobject_cast<ZLineEdit *>(ctrl.pControl)) {
        vec << lineEdit->text().toFloat();
    } else if (ZVecEditor *lineEdit = qobject_cast<ZVecEditor *>(ctrl.pControl)) {
        vec = lineEdit->text();
    }
    for (int i = 0; i < vec.size(); i++) {
        QString key = curve_util::getCurveKey(i);
        if (newVal.contains(key) && !keys.contains(key))
            continue;
        if (!newVal.contains(key) || (!newVal[key].visible && newVal[key].points.size() < 2))
            newVal[key] = CURVE_DATA();

        bool visible = keys.contains(key);
        getDelfCurveData(newVal[key], vec.at(i), visible, key);
        curve_util::updateRange(newVal);
    }

    AppHelper::socketEditFinished(QVariant::fromValue(newVal), m_idx, ctrl.m_viewIdx);
    updateTimelineKeys(newVal);
}

void ZenoPropPanel::delKeyFrame(const _PANEL_CONTROL &ctrl, const QStringList &keys) 
{
    CURVES_DATA newVal;
    if (ctrl.m_viewIdx.data(ROLE_PARAM_VALUE).canConvert<CURVES_DATA>())
        newVal = ctrl.m_viewIdx.data(ROLE_PARAM_VALUE).value<CURVES_DATA>();
    ZenoMainWindow *mainWin = zenoApp->getMainWindow();
    ZASSERT_EXIT(mainWin);
    ZTimeline *timeline = mainWin->timeline();
    ZASSERT_EXIT(timeline);
    int emptySize = 0;
    for (auto &curve : newVal) {
        QString key = curve.key;
        if (curve.visible == false && curve.points.size() < 2) {
            emptySize++;
            continue;
        }
        if (!keys.contains(key))
            continue;
        for (int i = 0; i < curve.points.size(); i++) {
            int x = curve.points.at(i).point.x();
            if (x == timeline->value()) {
                curve.points.remove(i);
                break;
            }
        }
        if (curve.points.isEmpty()) {
            emptySize++;
            if (ZVecEditor *lineEdit = qobject_cast<ZVecEditor *>(ctrl.pControl)) 
            {
                curve = CURVE_DATA();
                UI_VECTYPE vec = lineEdit->text();
                int idx = key == "x" ? 0 : key == "y" ? 1 : key == "z" ? 2 : 3;
                if (vec.size() > idx)
                    getDelfCurveData(curve, vec.at(idx), false, key);
            }
        }
    }
    QVariant val;
    if (emptySize == newVal.size()) 
    {
        if (ZVecEditor *lineEdit = qobject_cast<ZVecEditor *>(ctrl.pControl)) {
            val = QVariant::fromValue(lineEdit->text());
        } else if (ZLineEdit *lineEdit = qobject_cast<ZLineEdit *>(ctrl.pControl)) {
            val = QVariant::fromValue(lineEdit->text().toFloat());
        }
        newVal = CURVES_DATA();
    }
    else 
    {
        val = QVariant::fromValue(newVal);
    }
    AppHelper::socketEditFinished(val, m_idx, ctrl.m_viewIdx);
    updateTimelineKeys(newVal);
}

void ZenoPropPanel::editKeyFrame(const _PANEL_CONTROL &ctrl, const QStringList &keys) 
{
    ZQwtCurveMapEditor* pEditor = new ZQwtCurveMapEditor(true);
    connect(pEditor, &ZQwtCurveMapEditor::finished, this, [=](int result) {
        CURVES_DATA newCurves = pEditor->curves();
    CURVES_DATA val;
    if (ctrl.m_viewIdx.data(ROLE_PARAM_VALUE).canConvert<CURVES_DATA>())
        val = ctrl.m_viewIdx.data(ROLE_PARAM_VALUE).value<CURVES_DATA>();
    QVariant newVal;
    if (!newCurves.isEmpty() || val.size() != keys.size()) {
        for (const QString& key : keys) {
            if (newCurves.contains(key))
                val[key] = newCurves[key];
            else {
                val[key] = CURVE_DATA();
                if (ZVecEditor* lineEdit = qobject_cast<ZVecEditor*>(ctrl.pControl)) {
                    UI_VECTYPE vec = lineEdit->text();
                    int idx = key == "x" ? 0 : key == "y" ? 1 : key == "z" ? 2 : 3;
                    if (vec.size() > idx)
                        getDelfCurveData(val[key], vec.at(idx), false, key);
                }
            }
        }
        newVal = QVariant::fromValue(val);
    }
    else
    {
        if (ZLineEdit* lineEdit = qobject_cast<ZLineEdit*>(ctrl.pControl)) {
            newVal = QVariant::fromValue(lineEdit->text().toFloat());
        }
        else if (ZVecEditor* lineEdit = qobject_cast<ZVecEditor*>(ctrl.pControl)) {
            newVal = QVariant::fromValue(lineEdit->text());
        }
        val = CURVES_DATA();
    }
    AppHelper::socketEditFinished(newVal, m_idx, ctrl.m_viewIdx);
    updateTimelineKeys(val);
    });
    
    CURVES_DATA curves = getCurvesData(ctrl.m_viewIdx, keys);
    if (curves.size() > 1)
        curve_util::updateRange(curves);
    pEditor->setAttribute(Qt::WA_DeleteOnClose);
    pEditor->addCurves(curves);
    CURVES_MODEL models = pEditor->getModel();
    for (auto model :models) {
        for (int i = 0; i < model->rowCount(); i++) {
            model->setData(model->index(i, 0), true, ROLE_LOCKX);
        }
    }
    pEditor->exec();
}

void ZenoPropPanel::clearKeyFrame(const _PANEL_CONTROL& ctrl, const QStringList& keys)
{
    CURVES_DATA val;
    if (ctrl.m_viewIdx.data(ROLE_PARAM_VALUE).canConvert<CURVES_DATA>())
        val = ctrl.m_viewIdx.data(ROLE_PARAM_VALUE).value<CURVES_DATA>();
    int emptySize = 0;

    for (auto &curve : val) {
        QString key = curve.key;
        if (keys.contains(key)) {
            emptySize++;
            if (ZVecEditor* lineEdit = qobject_cast<ZVecEditor*>(ctrl.pControl))
            {
                curve = CURVE_DATA();
                UI_VECTYPE vec = lineEdit->text();
                int idx = key == "x" ? 0 : key == "y" ? 1 : key == "z" ? 2 : 3;
                if (vec.size() > idx)
                {
                    getDelfCurveData(curve, vec.at(idx), false, key);
                }
            }
        }
        else if (curve.visible == false && curve.points.size() < 2)
        {
            emptySize++;
        }
    }
    QVariant newVal;
    if (emptySize == val.size())
    {
        if (ZVecEditor* lineEdit = qobject_cast<ZVecEditor*>(ctrl.pControl)) {
            newVal = QVariant::fromValue(lineEdit->text());
        }
        else if (ZLineEdit* lineEdit = qobject_cast<ZLineEdit*>(ctrl.pControl)) {
            newVal = QVariant::fromValue(lineEdit->text().toFloat());
        }
        val = CURVES_DATA();
    }
    else
    {
        newVal = QVariant::fromValue(val);
    }
    AppHelper::socketEditFinished(newVal, m_idx, ctrl.m_viewIdx);
    updateTimelineKeys(val);
}

int ZenoPropPanel::getKeyFrameSize(const CURVES_DATA &curves)
{
    int size = 0;
    ZenoMainWindow *mainWin = zenoApp->getMainWindow();
    ZASSERT_EXIT(mainWin, false);
    ZTimeline *timeline = mainWin->timeline();
    ZASSERT_EXIT(timeline, false);
    int frame = timeline->value();
    for (auto curve : curves) {
        for (const auto &p : curve.points) {
            int x = p.point.x();
            if ((x == frame) && curve.visible) {
                size++;
                break;
            }
        }
    }
    return size;
}

QStringList ZenoPropPanel::getKeys(const QObject *obj, const _PANEL_CONTROL &ctrl) 
{
    QStringList keys;
    if (ZLineEdit *lineEdit = qobject_cast<ZLineEdit *>(ctrl.pControl))     //control float
    {
        keys << "x";
    } else if (ctrl.pLabel == obj) 
    {  //control label
        if (ctrl.m_viewIdx.data(ROLE_PARAM_VALUE).canConvert<UI_VECTYPE>()) {
            UI_VECTYPE vec = ctrl.m_viewIdx.data(ROLE_PARAM_VALUE).value<UI_VECTYPE>();
            for (int i = 0; i < vec.size(); i++) {
                QString key = curve_util::getCurveKey(i);
                if (!key.isEmpty())
                    keys << key;
            }
        } else if (ctrl.m_viewIdx.data(ROLE_PARAM_VALUE).canConvert<CURVES_DATA>()) {
            CURVES_DATA val = ctrl.m_viewIdx.data(ROLE_PARAM_VALUE).value<CURVES_DATA>();
            keys << val.keys();
        }
    } else if (ZVecEditor *vecEdit = qobject_cast<ZVecEditor *>(ctrl.pControl)) //control vec
    {
        int idx = vecEdit->getCurrentEditor();
        QString key = curve_util::getCurveKey(idx);
        if (!key.isEmpty())
            keys << key;
    }
    return keys;
}

CURVES_DATA ZenoPropPanel::getCurvesData(const QPersistentModelIndex &perIdx, const QStringList &keys) {
    CURVES_DATA val;
    if (perIdx.data(ROLE_PARAM_VALUE).canConvert<CURVES_DATA>())
        val = perIdx.data(ROLE_PARAM_VALUE).value<CURVES_DATA>();
    CURVES_DATA curves;
    for (auto key : keys) {
        if (val.contains(key)) {
            curves[key] = val[key];
        }
    }
    return curves;
}
void ZenoPropPanel::updateTimelineKeys(const CURVES_DATA &curves) 
{
    ZenoMainWindow *mainWin = zenoApp->getMainWindow();
    ZASSERT_EXIT(mainWin);
    ZTimeline *timeline = mainWin->timeline();
    ZASSERT_EXIT(timeline);
    QVector<int> keys = m_idx.data(ROLE_KEYFRAMES).value<QVector<int>>();
    timeline->updateKeyFrames(keys);
}

void ZenoPropPanel::onUpdateFrame(QWidget* pContrl, int nFrame, QVariant val)
{
    QVariant newVal = val;
    if (!AppHelper::getCurveValue(newVal))
        return;
    //vec
    if (ZVecEditor* pVecEdit = qobject_cast<ZVecEditor*>(pContrl))
    {
        pVecEdit->setVec(newVal, pVecEdit->isFloat());
        QVector<QString> properties = AppHelper::getKeyFrameProperty(val);
        pVecEdit->updateProperties(properties);
    }
    else if (QLineEdit* pLineEdit = qobject_cast<QLineEdit*>(pContrl))
    {
        QString text = UiHelper::variantToString(newVal);
        pLineEdit->setText(text);
        QVector<QString> properties = AppHelper::getKeyFrameProperty(val);
        pLineEdit->setProperty(g_setKey, properties.first());
        pLineEdit->style()->unpolish(pLineEdit);
        pLineEdit->style()->polish(pLineEdit);
        pLineEdit->update();
    }
}
