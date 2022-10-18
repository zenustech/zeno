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
#include <zenoui/comctrl/dialog/curvemap/zcurvemapeditor.h>
#include <zenoui/comctrl/dialog/zenoheatmapeditor.h>
#include "zenomainwindow.h"


ZenoPropPanel::ZenoPropPanel(QWidget* parent)
    : QWidget(parent)
    , m_bReentry(false)
{
    QVBoxLayout* pVLayout = new QVBoxLayout;
    pVLayout->setContentsMargins(QMargins(0, 0, 0, 0));
    setLayout(pVLayout);
    setFocusPolicy(Qt::ClickFocus);

    QPalette palette = this->palette();
    palette.setBrush(QPalette::Window, QColor("#2D3239"));
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
    m_groups.clear();
    update();
}

void ZenoPropPanel::reset(IGraphsModel* pModel, const QModelIndex& subgIdx, const QModelIndexList& nodes, bool select)
{
    clearLayout();
    QVBoxLayout *pMainLayout = qobject_cast<QVBoxLayout *>(this->layout());

    if (!pModel || !select || nodes.isEmpty())
    {
        update();
        return;
    }

    connect(pModel, &IGraphsModel::_dataChanged, this, &ZenoPropPanel::onDataChanged);
    connect(pModel, &IGraphsModel::_rowsRemoved, this, [=]() {
        clearLayout();
    });
    connect(pModel, &IGraphsModel::modelClear, this, [=]() {
        clearLayout();
    });

    m_subgIdx = subgIdx;
    m_idx = nodes[0];

    auto box = inputsBox(pModel, subgIdx, nodes);
    if (box)
    {
        pMainLayout->addWidget(box);
    }

    box = paramsBox(pModel, subgIdx, nodes);
    if (box)
    {
        pMainLayout->addWidget(box);
    }

    pMainLayout->addStretch();
    pMainLayout->setSpacing(0);

    onInputsCheckUpdate();
    onParamsCheckUpdate();

    update();
}

ZExpandableSection* ZenoPropPanel::paramsBox(IGraphsModel* pModel, const QModelIndex& subgIdx, const QModelIndexList& nodes)
{
    if (nodes.isEmpty())
        return nullptr;

    PARAMS_INFO params = m_idx.data(ROLE_PARAMETERS).value<PARAMS_INFO>();    if (params.isEmpty())
        return nullptr;

    ZExpandableSection* pParamsBox = new ZExpandableSection(tr("NODE PARAMETERS"));
    pParamsBox->setObjectName(tr("NODE PARAMETERS"));
    QGridLayout* pLayout = new QGridLayout;
    pLayout->setContentsMargins(10, 15, 0, 15);
    pLayout->setColumnStretch(0, 1);
    pLayout->setColumnStretch(1, 3);
    pLayout->setSpacing(10);
    pParamsBox->setContentLayout(pLayout);
    return pParamsBox;
}

ZExpandableSection* ZenoPropPanel::inputsBox(IGraphsModel* pModel, const QModelIndex& subgIdx, const QModelIndexList& nodes)
{
    ZASSERT_EXIT(m_idx.isValid(), nullptr);
    INPUT_SOCKETS inputs = m_idx.data(ROLE_INPUTS).value<INPUT_SOCKETS>();
    if (inputs.isEmpty())
        return nullptr;

    const QString& groupName = tr("SOCKET IN");
    ZExpandableSection* pInputsBox = new ZExpandableSection(groupName);
    pInputsBox->setObjectName(groupName);
    QGridLayout* pLayout = new QGridLayout;
    pLayout->setContentsMargins(10, 15, 0, 15);
    pLayout->setColumnStretch(0, 1);
    pLayout->setColumnStretch(1, 3);
    pLayout->setSpacing(10);
    pInputsBox->setContentLayout(pLayout);
    return pInputsBox;
}

void ZenoPropPanel::onInputsCheckUpdate()
{
    ZASSERT_EXIT(m_idx.isValid());
    INPUT_SOCKETS inputs = m_idx.data(ROLE_INPUTS).value<INPUT_SOCKETS>();
    QMap<QString, CONTROL_DATA> ctrls;
    for (const QString& inSock : inputs.keys())
    {
        const INPUT_SOCKET& inSocket = inputs[inSock];
        CONTROL_DATA ctrl;
        ctrl.ctrl = inSocket.info.control;
        ctrl.name = inSock;
        ctrl.typeDesc = inSocket.info.type;
        ctrl.value = inSocket.info.defaultValue;
        ctrl.cbFunc = [=](QVariant newValue)
        {
            IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();
            ZASSERT_EXIT(pGraphsModel);

            const QString& nodeid = m_idx.data(ROLE_OBJID).toString();
            const INPUT_SOCKETS& inputs = m_idx.data(ROLE_INPUTS).value<INPUT_SOCKETS>();
            if (inputs.find(inSock) == inputs.end())
                return;

            const INPUT_SOCKET& inSocket = inputs[inSock];

            PARAM_UPDATE_INFO info;
            info.name = inSock;
            info.oldValue = inSocket.info.defaultValue;
            info.newValue = newValue;
            zeno::scope_exit se([this]() { m_bReentry = false; });
            m_bReentry = true;
            pGraphsModel->updateSocketDefl(nodeid, info, m_subgIdx, true);
        };
        ctrls.insert(inSock, ctrl);
    }
    onGroupCheckUpdated(tr("SOCKET IN"), ctrls);
}

void ZenoPropPanel::onParamsCheckUpdate()
{
    ZASSERT_EXIT(m_idx.isValid());
    const QString& nodeid = m_idx.data(ROLE_OBJID).toString();
    PARAMS_INFO params = m_idx.data(ROLE_PARAMETERS).value<PARAMS_INFO>();
    QMap<QString, CONTROL_DATA> ctrls;
    for (const QString& name : params.keys())
    {
        const PARAM_INFO& param = params[name];
        CONTROL_DATA ctrl;
        ctrl.ctrl = param.control;
        ctrl.name = name;
        ctrl.typeDesc = param.typeDesc;
        ctrl.value = param.value;
        ctrl.bkFrame = (CONTROL_INT == ctrl.ctrl || CONTROL_FLOAT == ctrl.ctrl || CONTROL_VEC == ctrl.ctrl); //temp: todo: kframe.

        ctrl.cbFunc = [=](QVariant newValue) {
            IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();
            ZASSERT_EXIT(pGraphsModel);
            zeno::scope_exit se([this]() { m_bReentry = false; });
            m_bReentry = true;

            PARAM_UPDATE_INFO info;
            info.oldValue = UiHelper::getParamValue(m_idx, name);
            info.newValue = newValue;
            info.name = name;
            if (info.oldValue != info.newValue)
                pGraphsModel->updateParamInfo(nodeid, info, m_subgIdx, true);
        };
        ctrls.insert(name, ctrl);
    }
    onGroupCheckUpdated(tr("NODE PARAMETERS"), ctrls);
}

void ZenoPropPanel::onGroupCheckUpdated(const QString& groupName, const QMap<QString, CONTROL_DATA>& ctrls)
{
    if (ctrls.isEmpty())
        return;

    PANEL_GROUP& group = m_groups[groupName];

    ZExpandableSection* pExpand = findChild<ZExpandableSection*>(groupName);
    ZASSERT_EXIT(pExpand);

    auto cbSwith = [=](bool bOn) {
        zenoApp->getMainWindow()->setInDlgEventLoop(bOn);   //deal with ubuntu dialog slow problem when update viewport.
    };

    QGridLayout* pLayout = qobject_cast<QGridLayout*>(pExpand->contentLayout());
    ZASSERT_EXIT(pLayout);
    for (CONTROL_DATA ctrldata : ctrls)
    {
        const QString& name = ctrldata.name;
        _PANEL_CONTROL& item = group[ctrldata.name];
        if (!zenoui::isMatchControl(ctrldata.ctrl, item.pControl))
        {
            if (item.pControl)
            {
                //remove the dismatch control
                pLayout->removeWidget(item.pControl);
                delete item.pControl;
                item.pControl = nullptr;
            }

            item.pControl = zenoui::createWidget(ctrldata.value, ctrldata.ctrl, ctrldata.typeDesc, ctrldata.cbFunc, cbSwith);
            if (!item.pControl)
                continue;

            int n = pLayout->rowCount();
            if (!item.pLabel)
            {
                item.pLabel = new QLabel(ctrldata.name);
                item.pLabel->setProperty("cssClass", "proppanel");
                pLayout->addWidget(item.pLabel, n, 0, Qt::AlignLeft);
            }
            pLayout->addWidget(item.pControl, n, 1);
        }
        zenoui::updateValue(item.pControl, ctrldata.value);
    }
}

void ZenoPropPanel::onDataChanged(const QModelIndex& subGpIdx, const QModelIndex& idx, int role)
{
    //may be called frequently
    if (m_subgIdx != subGpIdx || m_idx != idx || m_bReentry)
        return;

    QLayout* pLayout = this->layout();
    if (!pLayout || pLayout->isEmpty())
        return;

    IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
    if (!pModel)
        return;

    if (role == ROLE_PARAMETERS)
    {
        onParamsCheckUpdate();
    }
    else if (role == ROLE_INPUTS)
    {
        onInputsCheckUpdate();
    }
    else
    {
        //other custom ui role.
        //onCustomUiUpdate();
    }
}
