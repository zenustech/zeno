#include <QtWidgets>
#include "nodeeditor/gv/zenonode.h"
#include "nodeeditor/gv/groupnode.h"
#include "zenoapplication.h"
#include "nodeeditor/gv/zenosubgraphscene.h"
#include "nodeeditor/gv/zenonewmenu.h"
#include "nodeeditor/gv/zenolink.h"
#include "model/graphsmanager.h"
#include "util/uihelper.h"
#include "nodeeditor/gv/zveceditoritem.h"
#include "nodeeditor/gv/zenogvhelper.h"
#include "variantptr.h"
#include "zassert.h"


static void dumpToClipboard(const QString& copyInfo)
{
    QMimeData* pMimeData = new QMimeData;
    pMimeData->setText(copyInfo);
    QApplication::clipboard()->setMimeData(pMimeData);
}

bool sceneMenuEvent(
    ZenoSubGraphScene* pScene,
    const QPointF& pos,
    const QPointF& scenePos,
    const QList<QGraphicsItem*>& seledItems,
    const QList<QGraphicsItem*>& items
    )
{
    QSet<ZenoNode*> nodeSets, nodeSelections;
    QSet<ZenoFullLink*> seledLinks;

    ZenoSocketItem* pSelSocket = nullptr;
    ZGraphicsNetLabel* pNetLabel = nullptr;
    QGraphicsItem* pFocusItem = nullptr;

    ZASSERT_EXIT(pScene, false);

    for (QGraphicsItem* pItem : seledItems)
    {
        if (ZenoNode* pNode = qgraphicsitem_cast<ZenoNode*>(pItem))
        {
            nodeSelections.insert(pNode);
        }
        else if (ZenoFullLink* pLink = qgraphicsitem_cast<ZenoFullLink*>(pItem))
        {
            seledLinks.insert(pLink);
        }
    }

    for (QGraphicsItem* pItem : items)
    {
        if (GroupNode* pGroup = dynamic_cast<GroupNode*>(pItem))
        {
            continue;
        }
        if (ZenoNode* pNode = qgraphicsitem_cast<ZenoNode*>(pItem))
        {
            nodeSets.insert(pNode);
        }
        else if (ZGraphicsNetLabel* _netLabel = qgraphicsitem_cast<ZGraphicsNetLabel*>(pItem))
        {
            pNetLabel = _netLabel;
        }
        pFocusItem = pItem;
    }

    if (nodeSets.size() == 1)
    {
        //send to scene/ZenoNode.
        ZenoNode* pNode = *nodeSets.begin();

        QModelIndex selParam;

        //check socket selection.
        for (QGraphicsItem* pItem : items)
        {
            if (ZSocketPlainTextItem* pSocketText = qgraphicsitem_cast<ZSocketPlainTextItem*>(pItem))
            {
                selParam = pNode->getSocketIndex(pSocketText, true);
                break;
            }
            else if (ZenoSocketItem* pSocketItem= qgraphicsitem_cast<ZenoSocketItem*>(pItem))
            {
                selParam = pSocketItem->paramIndex();
                break;
            }
        }

        if (selParam.isValid())
        {
            bool bInput = selParam.data(ROLE_ISINPUT).toBool();
            QString paramName = selParam.data(ROLE_PARAM_NAME).toString();
            int type = selParam.data(ROLE_PARAM_TYPE).toInt();

            QMenu* socketMenu = new QMenu;

            //check whether it's a vector param.
            if (type == zeno::Param_Vec2i || type == zeno::Param_Vec2f) {
                QMenu* pCopyElem = new QMenu(socketMenu);
                pCopyElem->setTitle(QObject::tr("copy vec param"));

                QAction* copy_x = new QAction(QObject::tr("copy vec.x"));
                QAction* copy_y = new QAction(QObject::tr("copy vec.y"));

                QObject::connect(copy_x, &QAction::triggered, [=]() {
                    QString str = UiHelper::getNaiveParamPath(selParam, 0);
                    QString refExpression = QString("ref(\"%1\")").arg(str);
                    dumpToClipboard(refExpression);
                });

                QObject::connect(copy_y, &QAction::triggered, [=]() {
                    QString str = UiHelper::getNaiveParamPath(selParam, 1);
                    QString refExpression = QString("ref(\"%1\")").arg(str);
                    dumpToClipboard(refExpression);
                });

                pCopyElem->addAction(copy_x);
                pCopyElem->addAction(copy_y);
                socketMenu->addAction(pCopyElem->menuAction());
            }
            else if (type == zeno::Param_Vec3i || type == zeno::Param_Vec3f) {
                QMenu* pCopyElem = new QMenu(socketMenu);
                pCopyElem->setTitle(QObject::tr("copy vec param"));

                QAction* copy_x = new QAction(QObject::tr("copy vec.x"));
                QAction* copy_y = new QAction(QObject::tr("copy vec.y"));
                QAction* copy_z = new QAction(QObject::tr("copy vec.z"));

                QObject::connect(copy_x, &QAction::triggered, [=]() {
                    QString str = UiHelper::getNaiveParamPath(selParam, 0);
                    QString refExpression = QString("ref(\"%1\")").arg(str);
                    dumpToClipboard(refExpression);
                });

                QObject::connect(copy_y, &QAction::triggered, [=]() {
                    QString str = UiHelper::getNaiveParamPath(selParam, 1);
                    QString refExpression = QString("ref(\"%1\")").arg(str);
                    dumpToClipboard(refExpression);
                });

                QObject::connect(copy_z, &QAction::triggered, [=]() {
                    QString str = UiHelper::getNaiveParamPath(selParam, 2);
                    QString refExpression = QString("ref(\"%1\")").arg(str);
                    dumpToClipboard(refExpression);
                });


                pCopyElem->addAction(copy_x);
                pCopyElem->addAction(copy_y);
                pCopyElem->addAction(copy_z);
                socketMenu->addAction(pCopyElem->menuAction());
            }
            else if (type == zeno::Param_Vec4i || type == zeno::Param_Vec4f) {
                QMenu* pCopyElem = new QMenu(socketMenu);
                pCopyElem->setTitle(QObject::tr("copy vec param"));

                QAction* copy_x = new QAction(QObject::tr("copy vec.x"));
                QAction* copy_y = new QAction(QObject::tr("copy vec.y"));
                QAction* copy_z = new QAction(QObject::tr("copy vec.z"));
                QAction* copy_w = new QAction(QObject::tr("copy vec.w"));

                QObject::connect(copy_x, &QAction::triggered, [=]() {
                    QString str = UiHelper::getNaiveParamPath(selParam, 0);
                    QString refExpression = QString("ref(\"%1\")").arg(str);
                    dumpToClipboard(refExpression);
                });

                QObject::connect(copy_y, &QAction::triggered, [=]() {
                    QString str = UiHelper::getNaiveParamPath(selParam, 1);
                    QString refExpression = QString("ref(\"%1\")").arg(str);
                    dumpToClipboard(refExpression);
                });

                QObject::connect(copy_z, &QAction::triggered, [=]() {
                    QString str = UiHelper::getNaiveParamPath(selParam, 2);
                    QString refExpression = QString("ref(\"%1\")").arg(str);
                    dumpToClipboard(refExpression);
                });

                QObject::connect(copy_w, &QAction::triggered, [=]() {
                    QString str = UiHelper::getNaiveParamPath(selParam, 3);
                    QString refExpression = QString("ref(\"%1\")").arg(str);
                    dumpToClipboard(refExpression);
                });

                pCopyElem->addAction(copy_x);
                pCopyElem->addAction(copy_y);
                pCopyElem->addAction(copy_z);
                pCopyElem->addAction(copy_w);
                socketMenu->addAction(pCopyElem->menuAction());
            }
            else {
            }

            //paste action for editable param
            if (type == zeno::Param_Float || 
                type == zeno::Param_Int || 
                type == zeno::Param_String)
            {
                const QMimeData* pMimeData_ = QApplication::clipboard()->mimeData();
                if (pMimeData_ && pMimeData_->text().startsWith("ref("))
                {
                    QAction* pasteRef = new QAction(QObject::tr("Paste Reference"));
                    QObject::connect(pasteRef, &QAction::triggered, [=]() {
                        const QMimeData* pMimeData = QApplication::clipboard()->mimeData();
                        if (pMimeData) {
                            QString refExp = pMimeData->text();
                            UiHelper::qIndexSetData(selParam, refExp, ROLE_PARAM_VALUE);
                        }
                        });
                    socketMenu->addAction(pasteRef);
                }
            }

            if (bInput) {

                //input socket menu
                QAction* pCopyRef = new QAction(QObject::tr("Copy Param Reference"));
                QObject::connect(pCopyRef, &QAction::triggered, [=]() {
                    QModelIndex nodeIdx = selParam.data(ROLE_NODE_IDX).toModelIndex();
                    if (nodeIdx.isValid() && nodeIdx.data(ROLE_CLASS_NAME) == "SubInput")
                    {
                        const QString& paramName = selParam.data(ROLE_PARAM_NAME).toString();
                        QString subgName, nodename, paramPath;
                        //TODO: deprecated.
                        QString str = selParam.data(ROLE_OBJPATH).toString();
                        UiHelper::getSocketInfo(str, subgName, nodename, paramPath);
                        if (paramName == "port") {
                            QString refExpression = QString("ref(%1/_IN_port)").arg(nodename);
                            dumpToClipboard(refExpression);
                            return;
                        }
                        else if (paramName == "hasValue") {
                            QString refExpression = QString("ref(%1/_IN_hasValue)").arg(nodename);
                            dumpToClipboard(refExpression);
                            return;
                        }
                    }

                    QString str = UiHelper::getNaiveParamPath(selParam);
                    QString refExpression = QString("ref(\"%1\")").arg(str);
                    dumpToClipboard(refExpression);
                });
                socketMenu->addAction(pCopyRef);
                //TODO: command params
#if 0
                IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
                ZASSERT_EXIT(pModel, false);
                const QString& path = selParam.data(ROLE_OBJPATH).toString();
                const QKeyList<QString, CommandParam>& params = pModel->commandParams();
                if (!params.contains(path))
                {
                    QAction* pCreateCommParam = new QAction(QObject::tr("Create Command Param"));
                    socketMenu->addAction(pCreateCommParam);
                    QObject::connect(pCreateCommParam, &QAction::triggered, [=]() {
                        CommandParam val;
                        val.name = paramName;
                        val.value = selParam.data(ROLE_PARAM_VALUE);
                        if (!pModel->addCommandParam(path, val))
                        {
                            QMessageBox::warning(nullptr, QObject::tr("Create Command Param"), QObject::tr("Create Command Param Failed!"));
                        }
                     });
                }
                else
                {
                    CommandParam command = params[path];
                    QAction* pDelCommParam = new QAction(QObject::tr("Delete Command Param"));
                    socketMenu->addAction(pDelCommParam);
                    QObject::connect(pDelCommParam, &QAction::triggered, [=]() {
                        if (QMessageBox::question(nullptr, QObject::tr("Delete Command Param"), QObject::tr("Delete %1 Command Param").arg(command.name)) == QMessageBox::Yes)
                        {
                            pModel->removeCommandParam(path);
                        }
                     });
                }
#endif
            }

            socketMenu->exec(QCursor::pos());
            socketMenu->deleteLater();
            return true;
        }
    }
    else
    {
        if (pFocusItem)
        {
            //dispatch to item.
            return false;
        }

        GraphModel* pGraphM = pScene->getGraphModel();
        ZASSERT_EXIT(pGraphM, false);

        zeno::NodeCates cates = zenoApp->graphsManager()->getCates();
        auto m_menu = new ZenoNewnodeMenu(pGraphM, cates, scenePos);
        m_menu->setEditorFocus();
        m_menu->exec(QCursor::pos());
        m_menu->deleteLater();
        return true;
    }
    return false;
}