#include "zenonodebase.h"
#include "zenosubgraphscene.h"
#include "uicommon.h"
#include "control/common_id.h"
#include "nodeeditor/gv/zenoparamnameitem.h"
#include "nodeeditor/gv/zenoparamwidget.h"
#include "util/uihelper.h"
#include "model/GraphsTreeModel.h"
#include "model/graphsmanager.h"
#include "model/parammodel.h"
#include <zeno/utils/logger.h>
#include <zeno/utils/scope_exit.h>
#include "style/zenostyle.h"
#include "widgets/zveceditor.h"
#include "variantptr.h"
#include "curvemap/zcurvemapeditor.h"
#include "zenoapplication.h"
#include "zenomainwindow.h"
#include "nodeeditor/gv/zenographseditor.h"
#include "util/log.h"
#include "zenosubgraphview.h"
#include "dialog/zenoheatmapeditor.h"
#include "nodeeditor/gv/zitemfactory.h"
#include "zvalidator.h"
#include "zenonewmenu.h"
#include "util/apphelper.h"
#include "viewport/viewportwidget.h"
#include "viewport/displaywidget.h"
#include "nodeeditor/gv/zgraphicstextitem.h"
#include "nodeeditor/gv/zenogvhelper.h"
#include "iotags.h"
#include "groupnode.h"
#include "dialog/zeditparamlayoutdlg.h"
#include "settings/zenosettingsmanager.h"
#include "nodeeditor/gv/zveceditoritem.h"
#include "nodeeditor/gv/zdictsocketlayout.h"
#include "zassert.h"
#include "widgets/ztimeline.h"
#include "socketbackground.h"
#include "statusgroup.h"
#include "statusbutton.h"
#include "model/assetsmodel.h"
#include <zeno/utils/helper.h>


ZenoNodeBase::ZenoNodeBase(const NodeUtilParam &params, QGraphicsItem *parent)
    : _base(parent)
    , m_renderParams(params)
    , m_bMoving(false)
    , m_bUIInited(false)
    , m_groupNode(nullptr)
    , m_bVisible(true)
{
    setFlags(ItemIsMovable | ItemIsSelectable);
    setAcceptHoverEvents(true);
}

ZenoNodeBase::~ZenoNodeBase()
{
}

void ZenoNodeBase::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
    if (isSelected())
    {
        _drawBorderWangStyle(painter);
    }
}

void ZenoNodeBase::_drawBorderWangStyle(QPainter* painter)
{
	//draw inner border
	painter->setRenderHint(QPainter::Antialiasing, true);
    QColor baseColor = /*m_bError ? QColor(200, 84, 79) : */QColor(255, 100, 0);
	QColor borderClr(baseColor);
	borderClr.setAlphaF(0.2);
    qreal innerBdrWidth = ZenoStyle::scaleWidth(6);
	QPen pen(borderClr, innerBdrWidth);
	pen.setJoinStyle(Qt::MiterJoin);
	painter->setPen(pen);

	QRectF rc = boundingRect();
	qreal offset = innerBdrWidth / 2; //finetune
	rc.adjust(-offset, -offset, offset, offset);
	QPainterPath path = UiHelper::getRoundPath(rc, m_renderParams.headerBg.lt_radius, m_renderParams.headerBg.rt_radius, m_renderParams.bodyBg.lb_radius, m_renderParams.bodyBg.rb_radius, true);
	painter->drawPath(path);

    //draw outter border
    qreal outterBdrWidth = ZenoStyle::scaleWidth(2);
    pen.setWidthF(outterBdrWidth);
    pen.setColor(baseColor);
	painter->setPen(pen);
    offset = outterBdrWidth;
    rc.adjust(-offset, -offset, offset, offset);
    path = UiHelper::getRoundPath(rc, m_renderParams.headerBg.lt_radius, m_renderParams.headerBg.rt_radius, m_renderParams.bodyBg.lb_radius, m_renderParams.bodyBg.rb_radius, true);
    painter->drawPath(path);
}


int ZenoNodeBase::type() const
{
    return Type;
}

void ZenoNodeBase::initUI(const QModelIndex& index)
{
    ZASSERT_EXIT(index.isValid());
    m_index = QPersistentModelIndex(index);

    QPointF pos = m_index.data(ROLE_OBJPOS).toPointF();
    const QString &id = m_index.data(ROLE_NODE_NAME).toString();
    setPos(pos);
    initLayout();
    // setPos will send geometry, but it's not supposed to happend during initialization.
    setFlag(ItemSendsGeometryChanges);
    setFlag(ItemSendsScenePositionChanges);

    bool bCollasped = m_index.data(ROLE_COLLASPED).toBool();
    onCollaspeUpdated(bCollasped);

    m_bUIInited = true;
}

void ZenoNodeBase::updateWhole()
{
    ZGraphicsLayout::updateHierarchy(this);
    emit inSocketPosChanged();
    emit outSocketPosChanged();
}


void ZenoNodeBase::markNodeStatus(zeno::NodeRunStatus status)
{
    update();
}


ZenoSocketItem* ZenoNodeBase::getSocketItem(const QModelIndex& sockIdx, const QString keyName)
{
    return nullptr;
}

ZenoSocketItem* ZenoNodeBase::getNearestSocket(const QPointF& pos, bool bInput)
{
    return nullptr;
}

QModelIndex ZenoNodeBase::getSocketIndex(QGraphicsItem* uiitem, bool bSocketText) const
{
    return QModelIndex();
}

QPointF ZenoNodeBase::getSocketPos(const QModelIndex& sockIdx, const QString keyName)
{
    return QPointF(0, 0);
}

QString ZenoNodeBase::nodeId() const
{
    ZASSERT_EXIT(m_index.isValid(), "");
    return m_index.data(ROLE_NODE_NAME).toString();
}

QString ZenoNodeBase::nodeClass() const
{
    ZASSERT_EXIT(m_index.isValid(), "");
    return m_index.data(ROLE_CLASS_NAME).toString();
}

QPointF ZenoNodeBase::nodePos() const
{
    ZASSERT_EXIT(m_index.isValid(), QPointF());
    return m_index.data(ROLE_OBJPOS).toPointF();
}

void ZenoNodeBase::setMoving(bool isMoving)
{
    m_bMoving = isMoving;
}

void ZenoNodeBase::onZoomed()
{
}

void ZenoNodeBase::setGroupNode(GroupNode *pNode) 
{
    m_groupNode = pNode;
}

GroupNode *ZenoNodeBase::getGroupNode() 
{
    return m_groupNode;
}

ZenoGraphsEditor* ZenoNodeBase::getEditorViewByViewport(QWidget* pWidget)
{
    QWidget* p = pWidget;
    while (p)
    {
        if (ZenoGraphsEditor* pEditor = qobject_cast<ZenoGraphsEditor*>(p))
            return pEditor;
        p = p->parentWidget();
    }
    return nullptr;
}

void ZenoNodeBase::contextMenuEvent(QGraphicsSceneContextMenuEvent* event)
{
    auto graphsMgr = zenoApp->graphsManager();
    QPointF pos = event->pos();
    if (m_index.data(ROLE_NODETYPE) == zeno::Node_SubgraphNode)
    {
        scene()->clearSelection();
        this->setSelected(true);

        QMenu *nodeMenu = new QMenu;
        QAction *pCopy = new QAction("Copy");
        QAction *pDelete = new QAction("Delete");

        connect(pDelete, &QAction::triggered, this, [=]() {
            //pGraphsModel->removeNode(m_index.data(ROLE_NODE_NAME).toString(), m_subGpIndex, true);
        });

        nodeMenu->addAction(pCopy);
        nodeMenu->addAction(pDelete);

        QAction* propDlg = new QAction(tr("Custom Param"));
        nodeMenu->addAction(propDlg);
        connect(propDlg, &QAction::triggered, this, [=]() {
            ParamsModel* paramsM = QVariantPtr<ParamsModel>::asPtr(m_index.data(ROLE_PARAMS));

            ZenoSubGraphScene* pScene = qobject_cast<ZenoSubGraphScene*>(scene());
            ZASSERT_EXIT(pScene && !pScene->views().isEmpty());
            if (_ZenoSubGraphView* pView = qobject_cast<_ZenoSubGraphView*>(pScene->views().first()))
            {
                ZEditParamLayoutDlg dlg(paramsM->customParamModel(), pView);
                if (QDialog::Accepted == dlg.exec())
                {
                    zeno::ParamsUpdateInfo info = dlg.getEdittedUpdateInfo();
                    paramsM->resetCustomUi(dlg.getCustomUiInfo());
                    paramsM->batchModifyParams(info);
                }
            }
        });
        QAction* saveAsset = new QAction(tr("Save as asset"));
        nodeMenu->addAction(saveAsset);
        connect(saveAsset, &QAction::triggered, this, [=]() {
            QString name = m_index.data(ROLE_NODE_NAME).toString();
            AssetsModel* pModel = zenoApp->graphsManager()->assetsModel();
            if (pModel->getAssetGraph(name))
            {
                QMessageBox::warning(nullptr, tr("Save as asset"), tr("Asset %1 is existed").arg(name));
                return;
            }
            zeno::ZenoAsset asset;
            asset.info.name = name.toStdString();
            QString dirPath = QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation);
            QString path = dirPath + "/ZENO/assets/" + name + ".zda";
            path = QFileDialog::getSaveFileName(nullptr, "File to Open", path, "All Files(*);;");
            if (path.isEmpty())
                return;
            asset.info.path = path.toStdString();
            asset.info.majorVer = 1;
            asset.info.minorVer = 0;
            zeno::NodeData data = m_index.data(ROLE_NODEDATA).value<zeno::NodeData>();
            asset.object_inputs = data.customUi.inputObjs;
            asset.object_outputs = data.customUi.outputObjs;
            asset.primitive_inputs = zeno::customUiToParams(data.customUi.inputPrims);
            asset.primitive_outputs = data.customUi.outputPrims;
            asset.optGraph = data.subgraph;
            asset.m_customui = data.customUi;
            auto& assets = zeno::getSession().assets;
            assets->createAsset(asset);
            pModel->saveAsset(name);
        });

        nodeMenu->exec(QCursor::pos());
        nodeMenu->deleteLater();
    }
    else if (m_index.data(ROLE_NODETYPE) == zeno::Node_AssetInstance)
    {
        GraphModel* pSubgGraphM = m_index.data(ROLE_SUBGRAPH).value<GraphModel*>();
        ZASSERT_EXIT(pSubgGraphM);
        bool bLocked = pSubgGraphM->isLocked();
        QMenu* nodeMenu = new QMenu;
        QAction* pLock = new QAction(bLocked ? tr("UnLock") : tr("Lock"));
        nodeMenu->addAction(pLock);
        connect(pLock, &QAction::triggered, this, [=]() {
            pSubgGraphM->setLocked(!bLocked);
        });
        QAction* pEditParam = new QAction(tr("Custom Params"));
        nodeMenu->addAction(pEditParam);
        connect(pEditParam, &QAction::triggered, this, [=]() {
            ZenoGraphsEditor* pEditor = getEditorViewByViewport(event->widget());
            if (pEditor)
            {
                QString assetName = m_index.data(ROLE_CLASS_NAME).toString();
                pEditor->onAssetsCustomParamsClicked(assetName);
            }
        });
        nodeMenu->exec(QCursor::pos());
        nodeMenu->deleteLater();
    }
    else if (m_index.data(ROLE_CLASS_NAME).toString() == "BindMaterial")
    {
#if 0
        QAction* newSubGraph = new QAction(tr("Create Material Subgraph"));
        connect(newSubGraph, &QAction::triggered, this, [=]() {
            NodeParamModel* pNodeParams = QVariantPtr<NodeParamModel>::asPtr(m_index.data(ROLE_NODE_PARAMS));
            ZASSERT_EXIT(pNodeParams);
            const auto& paramIdx = pNodeParams->getParam(PARAM_INPUT, "mtlid");
            ZASSERT_EXIT(paramIdx.isValid());
            QString mtlid = paramIdx.data(ROLE_PARAM_VALUE).toString();
            if (!pGraphsModel->newMaterialSubgraph(m_subGpIndex, mtlid, this->pos() + QPointF(800, 0)))
                QMessageBox::warning(nullptr, tr("Info"), tr("Create material subgraph '%1' failed.").arg(mtlid));
        });
        QMenu *nodeMenu = new QMenu;
        nodeMenu->addAction(newSubGraph);
        nodeMenu->exec(QCursor::pos());
        nodeMenu->deleteLater();
#endif
    }
}

void ZenoNodeBase::mouseReleaseEvent(QGraphicsSceneMouseEvent* event)
{
    _base::mouseReleaseEvent(event);
    if (m_bMoving)
    {
        m_bMoving = false;
        QPointF newPos = this->scenePos();
        QPointF oldPos = m_index.data(ROLE_OBJPOS).toPointF();

        QAbstractItemModel* pModel = const_cast<QAbstractItemModel*>(m_index.model());
        GraphModel* model = qobject_cast<GraphModel*>(pModel);
        if (!model)
            return;

        if (newPos != oldPos)
        {
            model->setModelData(m_index, m_lastMoving, ROLE_OBJPOS);

            emit inSocketPosChanged();
            emit outSocketPosChanged();
            //emit nodePosChangedSignal();

            m_lastMoving = QPointF();

            //other selected items also need update model data
            for (QGraphicsItem *item : this->scene()->selectedItems()) {
                if (item == this || !dynamic_cast<ZenoNodeBase*>(item))
                    continue;
                ZenoNodeBase *pNode = dynamic_cast<ZenoNodeBase *>(item);
                model->setModelData(pNode->index(), pNode->scenePos(), ROLE_OBJPOS);
            }
        }
    }
}

QVariant ZenoNodeBase::itemChange(GraphicsItemChange change, const QVariant &value)
{
    if (!m_bUIInited)
        return value;

    if (change == QGraphicsItem::ItemSelectedHasChanged)
    {
        bool bSelected = isSelected();

        ZenoSubGraphScene *pScene = qobject_cast<ZenoSubGraphScene *>(scene());
        ZASSERT_EXIT(pScene, value);
        const QString& name = m_index.data(ROLE_NODE_NAME).toString();
        pScene->collectNodeSelChanged(name, bSelected);
    }
    else if (change == QGraphicsItem::ItemPositionChange)
    {
        m_bMoving = true;
        ZenoSubGraphScene* pScene = qobject_cast<ZenoSubGraphScene*>(scene());
        bool isSnapGrid = ZenoSettingsManager::GetInstance().getValue(zsSnapGrid).toBool();
        if (pScene) {
            if (isSnapGrid)
            {
                QPointF pos = value.toPointF();
                int x = pos.x(), y = pos.y();
                x = x - x % SCENE_GRID_SIZE;
                y = y - y % SCENE_GRID_SIZE;
                return QPointF(x, y);
            }
        }
        
    }
    else if (change == QGraphicsItem::ItemPositionHasChanged)
    {
        m_bMoving = true;
        m_lastMoving = value.toPointF();
        emit inSocketPosChanged();
        emit outSocketPosChanged();
        ZenoSubGraphScene* pScene = qobject_cast<ZenoSubGraphScene*>(scene());
        if (pScene) {
            pScene->onNodePositionChanged(this);
        }
    }
    else if (change == ItemScenePositionHasChanged)
    {
        emit inSocketPosChanged();
        emit outSocketPosChanged();
    }
    else if (change == ItemZValueHasChanged)
    {
        int type = m_index.data(ROLE_NODETYPE).toInt();
        if (type == zeno::Node_Group && zValue() != ZVALUE_BLACKBOARD)
        {
            setZValue(ZVALUE_BLACKBOARD);
        }
    }
    return value;
}

void ZenoNodeBase::onCollaspeBtnClicked()
{
}

void ZenoNodeBase::onCollaspeUpdated(bool)
{
}
