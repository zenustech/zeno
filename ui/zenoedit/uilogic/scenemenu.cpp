#include <QtWidgets>
#include "../nodesys/zenonode.h"
#include "zenoapplication.h"
#include "../nodesys/zenosubgraphscene.h"
#include "../nodesys/zenonewmenu.h"
#include "../nodesys/zenolink.h"
#include <zenomodel/include/graphsmanagment.h>
#include <zenomodel/include/command.h>
#include <zenomodel/include/uihelper.h>
#include <zenoui/comctrl/gv/zgraphicsnetlabel.h>
#include <zenoui/comctrl/gv/zveceditoritem.h>
#include <zenoui/comctrl/gv/zenogvhelper.h>
#include "variantptr.h"


static void onPasteSocketRefSlot(ZenoSubGraphScene* pScene, QModelIndex toIndex)
{
    ZASSERT_EXIT(pScene);
    const QMimeData* pMimeData = QApplication::clipboard()->mimeData();
    IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();
    ZASSERT_EXIT(pGraphsModel);

    auto clipboard = zenoApp->procClipboard();
    const QString& copiedParam = clipboard->getCopiedAddress();

    if (!copiedParam.isEmpty())
    {
        pScene->setProperty("link_label_info", "");
        pGraphsModel->beginTransaction(QObject::tr("add Link"));
        zeno::scope_exit sp([=]() { pGraphsModel->endTransaction(); });

        QModelIndex fromIndex = pGraphsModel->indexFromPath(copiedParam);
        if (fromIndex.isValid())
        {
            //remove the edge in inNode:inSock, if exists.
            int inProp = toIndex.data(ROLE_PARAM_SOCKPROP).toInt();
            if (inProp & SOCKPROP_DICTLIST_PANEL)
            {
                QString inSockType = toIndex.data(ROLE_PARAM_TYPE).toString();
                SOCKET_PROPERTY outProp = (SOCKET_PROPERTY)fromIndex.data(ROLE_PARAM_SOCKPROP).toInt();
                QString outSockType = fromIndex.data(ROLE_PARAM_TYPE).toString();
                QAbstractItemModel* pKeyObjModel =
                    QVariantPtr<QAbstractItemModel>::asPtr(toIndex.data(ROLE_VPARAM_LINK_MODEL));

                bool outSockIsContainer = false;
                if (inSockType == "list")
                {
                    outSockIsContainer = outSockType == "list";
                }
                else if (inSockType == "dict")
                {
                    const QModelIndex& fromNodeIdx = fromIndex.data(ROLE_NODE_IDX).toModelIndex();
                    const QString& outNodeCls = fromNodeIdx.data(ROLE_OBJNAME).toString();
                    const QString& outSockName = fromIndex.data(ROLE_PARAM_NAME).toString();
                    outSockIsContainer = outSockType == "dict" || (outNodeCls == "FuncBegin" && outSockName == "args");
                }

                //if outSock is a container, connects it as usual.
                if (outSockIsContainer)
                {
                    //legacy dict/list connection, and then we have to remove all inner dict key connection.
                    ZASSERT_EXIT(pKeyObjModel);
                    for (int r = 0; r < pKeyObjModel->rowCount(); r++)
                    {
                        const QModelIndex& keyIdx = pKeyObjModel->index(r, 0);
                        PARAM_LINKS links = keyIdx.data(ROLE_PARAM_LINKS).value<PARAM_LINKS>();
                        for (QPersistentModelIndex _linkIdx : links)
                        {
                            pGraphsModel->removeLink(_linkIdx, true);
                        }
                    }
                }
                else
                {
                    //check multiple links
                    QModelIndexList fromSockets;
                    //check selected nodes.
                    //model: ViewParamModel
                    QString paramName = fromIndex.data(ROLE_PARAM_NAME).toString();
                    QString paramType = fromIndex.data(ROLE_PARAM_TYPE).toString();

                    QString toSockName = toIndex.data(ROLE_OBJPATH).toString();

                    // link to inner dict key automatically.
                    int n = pKeyObjModel->rowCount();
                    pGraphsModel->addExecuteCommand(new DictKeyAddRemCommand(true, pGraphsModel, toIndex.data(ROLE_OBJPATH).toString(), n));
                    toIndex = pKeyObjModel->index(n, 0);
                }
            }
            if (inProp != SOCKPROP_MULTILINK)
            {
                QPersistentModelIndex linkIdx;
                const PARAM_LINKS& links = toIndex.data(ROLE_PARAM_LINKS).value<PARAM_LINKS>();
                if (!links.isEmpty())
                    linkIdx = links[0];
                if (linkIdx.isValid())
                    pGraphsModel->removeLink(linkIdx, true);
            }
            pGraphsModel->addLink(pScene->subGraphIndex(), fromIndex, toIndex, true);
        }
    }
}

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
    const QList<QGraphicsItem*>& items,
    const QModelIndex& subgIdx
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
        }

        if (selParam.isValid())
        {
            PARAM_CLASS coreCls = (PARAM_CLASS)selParam.data(ROLE_PARAM_CLASS).toInt();
            PARAM_CONTROL ctrl = (PARAM_CONTROL)selParam.data(ROLE_PARAM_CTRL).toInt();
            QString paramName = selParam.data(ROLE_PARAM_NAME).toString();
            QString type = selParam.data(ROLE_PARAM_TYPE).toString();

            QMenu* socketMenu = new QMenu;

            //check whether it's a vector param.
            if (type.startsWith("vec2")) {
                QMenu* pCopyElem = new QMenu(socketMenu);
                pCopyElem->setTitle(QObject::tr("copy vec param"));

                QAction* copy_x = new QAction(QObject::tr("copy vec.x"));
                QAction* copy_y = new QAction(QObject::tr("copy vec.y"));

                QObject::connect(copy_x, &QAction::triggered, [=]() {
                    QString str = UiHelper::getNaiveParamPath(selParam, 0);
                    QString refExpression = QString("ref(%1)").arg(str);
                    dumpToClipboard(refExpression);
                });

                QObject::connect(copy_y, &QAction::triggered, [=]() {
                    QString str = UiHelper::getNaiveParamPath(selParam, 1);
                    QString refExpression = QString("ref(%1)").arg(str);
                    dumpToClipboard(refExpression);
                });

                pCopyElem->addAction(copy_x);
                pCopyElem->addAction(copy_y);
                socketMenu->addAction(pCopyElem->menuAction());
            }
            else if (type.startsWith("vec3")) {
                QMenu* pCopyElem = new QMenu(socketMenu);
                pCopyElem->setTitle(QObject::tr("copy vec param"));

                QAction* copy_x = new QAction(QObject::tr("copy vec.x"));
                QAction* copy_y = new QAction(QObject::tr("copy vec.y"));
                QAction* copy_z = new QAction(QObject::tr("copy vec.z"));

                QObject::connect(copy_x, &QAction::triggered, [=]() {
                    QString str = UiHelper::getNaiveParamPath(selParam, 0);
                    QString refExpression = QString("ref(%1)").arg(str);
                    dumpToClipboard(refExpression);
                });

                QObject::connect(copy_y, &QAction::triggered, [=]() {
                    QString str = UiHelper::getNaiveParamPath(selParam, 1);
                    QString refExpression = QString("ref(%1)").arg(str);
                    dumpToClipboard(refExpression);
                });

                QObject::connect(copy_z, &QAction::triggered, [=]() {
                    QString str = UiHelper::getNaiveParamPath(selParam, 2);
                    QString refExpression = QString("ref(%1)").arg(str);
                    dumpToClipboard(refExpression);
                });


                pCopyElem->addAction(copy_x);
                pCopyElem->addAction(copy_y);
                pCopyElem->addAction(copy_z);
                socketMenu->addAction(pCopyElem->menuAction());
            }
            else if (type.startsWith("vec4")) {
                QMenu* pCopyElem = new QMenu(socketMenu);
                pCopyElem->setTitle(QObject::tr("copy vec param"));

                QAction* copy_x = new QAction(QObject::tr("copy vec.x"));
                QAction* copy_y = new QAction(QObject::tr("copy vec.y"));
                QAction* copy_z = new QAction(QObject::tr("copy vec.z"));
                QAction* copy_w = new QAction(QObject::tr("copy vec.w"));

                QObject::connect(copy_x, &QAction::triggered, [=]() {
                    QString str = UiHelper::getNaiveParamPath(selParam, 0);
                    QString refExpression = QString("ref(%1)").arg(str);
                    dumpToClipboard(refExpression);
                });

                QObject::connect(copy_y, &QAction::triggered, [=]() {
                    QString str = UiHelper::getNaiveParamPath(selParam, 1);
                    QString refExpression = QString("ref(%1)").arg(str);
                    dumpToClipboard(refExpression);
                });

                QObject::connect(copy_z, &QAction::triggered, [=]() {
                    QString str = UiHelper::getNaiveParamPath(selParam, 2);
                    QString refExpression = QString("ref(%1)").arg(str);
                    dumpToClipboard(refExpression);
                });

                QObject::connect(copy_w, &QAction::triggered, [=]() {
                    QString str = UiHelper::getNaiveParamPath(selParam, 3);
                    QString refExpression = QString("ref(%1)").arg(str);
                    dumpToClipboard(refExpression);
                });

                pCopyElem->addAction(copy_x);
                pCopyElem->addAction(copy_y);
                pCopyElem->addAction(copy_z);
                pCopyElem->addAction(copy_w);
                socketMenu->addAction(pCopyElem->menuAction());
            }
            else {
                if (PARAM_OUTPUT == coreCls) {
                    QString netlbl = selParam.data(ROLE_PARAM_NETLABEL).toString();
                    if (netlbl.isEmpty()) {
                        QAction* pCopyCreateLabel = new QAction(QObject::tr("Copy and Create Net Label"));
                        QObject::connect(pCopyCreateLabel, &QAction::triggered, [=]() {
                            //create a net label for socket param.
                            QString netlabel = UiHelper::getNaiveParamPath(selParam);
                            IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
                            pModel->addNetLabel(subgIdx, selParam, netlabel);
                            auto procClipbrd = zenoApp->procClipboard();
                            ZASSERT_EXIT(procClipbrd);
                            procClipbrd->setCopiedAddress(netlabel);
                        });
                        socketMenu->addAction(pCopyCreateLabel);
                    }
                    else {
                        QAction* pCopyLabel = new QAction(QObject::tr("Copy Net Label"));
                        QObject::connect(pCopyLabel, &QAction::triggered, [=]() {
                            const QString& netlabel = selParam.data(ROLE_PARAM_NETLABEL).toString();
                            ZASSERT_EXIT(!netlabel.isEmpty());
                            auto procClipbrd = zenoApp->procClipboard();
                            ZASSERT_EXIT(procClipbrd);
                            procClipbrd->setCopiedAddress(netlabel);
                        });

                        QAction* pDelNetLabel = new QAction(QObject::tr("Delete Net Label"));
                        QObject::connect(pDelNetLabel, &QAction::triggered, [=]() {
                            const QString& netlabel = selParam.data(ROLE_PARAM_NETLABEL).toString();
                            IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
                            ZASSERT_EXIT(pModel);
                            pModel->removeNetLabel(subgIdx, selParam);
                            auto procClipbrd = zenoApp->procClipboard();
                            procClipbrd->setCopiedAddress("");
                        });
 
                        socketMenu->addAction(pCopyLabel);
                        socketMenu->addAction(pDelNetLabel);
                    }
                }
            }

            //paste action for editable param
            if (type.startsWith("float") ||
                type.startsWith("int") ||
                type.startsWith("string"))
            {
                const QMimeData* pMimeData_ = QApplication::clipboard()->mimeData();
                if (pMimeData_ && pMimeData_->text().startsWith("ref("))
                {
                    QAction* pasteRef = new QAction(QObject::tr("Paste Reference"));
                    QObject::connect(pasteRef, &QAction::triggered, [=]() {
                        const QMimeData* pMimeData = QApplication::clipboard()->mimeData();
                        if (pMimeData) {
                            QString refExp = pMimeData->text();
                            IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
                            pModel->ModelSetData(selParam, refExp, ROLE_PARAM_VALUE);
                        }
                        });
                    socketMenu->addAction(pasteRef);
                }
            }

            if (PARAM_INPUT == coreCls) {

                //input socket menu
                QAction* pCopyRef = new QAction(QObject::tr("Copy Param Reference"));
                QObject::connect(pCopyRef, &QAction::triggered, [=]() {
                    QModelIndex nodeIdx = selParam.data(ROLE_NODE_IDX).toModelIndex();
                    if (nodeIdx.isValid() && nodeIdx.data(ROLE_OBJNAME) == "SubInput")
                    {
                        const QString& paramName = selParam.data(ROLE_PARAM_NAME).toString();
                        QString subgName, ident, paramPath;
                        QString str = selParam.data(ROLE_OBJPATH).toString();
                        UiHelper::getSocketInfo(str, subgName, ident, paramPath);
                        if (paramName == "port") {
                            QString refExpression = QString("ref(%1/_IN_port)").arg(ident);
                            dumpToClipboard(refExpression);
                            return;
                        }
                        else if (paramName == "hasValue") {
                            QString refExpression = QString("ref(%1/_IN_hasValue)").arg(ident);
                            dumpToClipboard(refExpression);
                            return;
                        }
                    }

                    QString str = UiHelper::getNaiveParamPath(selParam);
                    QString refExpression = QString("ref(%1)").arg(str);
                    dumpToClipboard(refExpression);
                });

                QString netlbl = selParam.data(ROLE_PARAM_NETLABEL).toString();
                if (netlbl.isEmpty()) {

                    auto procClipbrd = zenoApp->procClipboard();
                    ZASSERT_EXIT(procClipbrd, false);
                    const QString& lblName = procClipbrd->getCopiedAddress();
                    if (!lblName.isEmpty())
                    {
                        QAction* pPasteNetlbl = new QAction(QObject::tr("Paste Net Label"));
                        QObject::connect(pPasteNetlbl, &QAction::triggered, [=]() {
                            IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
                            ZASSERT_EXIT(pModel);
                            pModel->addNetLabel(subgIdx, selParam, lblName);
                        });
                        socketMenu->addAction(pPasteNetlbl);
                        socketMenu->addSeparator();
                    }

                    IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
                    ZASSERT_EXIT(pModel, false);

                    QStringList labels = pModel->dumpLabels(subgIdx);
                    if (!labels.isEmpty())
                    {
                        QMenu* pLabelsMenu = new QMenu(QObject::tr("Net Labels"));
                        for (auto lbl : labels)
                        {
                            QAction* pLblAction = new QAction(lbl);
                            QObject::connect(pLblAction, &QAction::triggered, [=]() {
                                IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
                                ZASSERT_EXIT(pModel);
                                pModel->addNetLabel(subgIdx, selParam, lbl);
                            });
                            pLabelsMenu->addAction(pLblAction);
                        }
                        socketMenu->addAction(pLabelsMenu->menuAction());
                        socketMenu->addSeparator();
                    }
                }
                else {
                    QAction* pRemoveNetlbl = new QAction(QObject::tr("Delete Net Label"));
                    QObject::connect(pRemoveNetlbl, &QAction::triggered, [=]() {
                        const QString& netlabel = selParam.data(ROLE_PARAM_NETLABEL).toString();
                        IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
                        ZASSERT_EXIT(pModel);
                        pModel->removeNetLabel(subgIdx, selParam);
                    });
                    socketMenu->addAction(pRemoveNetlbl);
                    socketMenu->addSeparator();
                }
                socketMenu->addAction(pCopyRef);
            }
            else if (PARAM_PARAM == coreCls) {
                int j;
                j = 0;
            }
            else if (PARAM_OUTPUT == coreCls) {
                
            }

            socketMenu->exec(QCursor::pos());
            socketMenu->deleteLater();
            return true;
        }
    }
    else
    {
        if (seledLinks.size() > 1)
        {
            //support links to transfer to net labels.
            QMenu* menu = new QMenu;
            QAction* pTransferToNetLabel = new QAction(QObject::tr("transfer to net label"));
            QObject::connect(pTransferToNetLabel, &QAction::triggered, [=]() {
                IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
                ZASSERT_EXIT(pModel);
                pModel->beginTransaction("transfer to net labels");
                zeno::scope_exit scope([=]() { pModel->endTransaction(); });
                for (ZenoFullLink* link : seledLinks) {
                    link->transferToNetLabel();
                }
            });

            QAction* pDelete = new QAction(QObject::tr("Delete"));
            QObject::connect(pDelete, &QAction::triggered, [=]() {
                IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
                ZASSERT_EXIT(pModel);
                pModel->beginTransaction("remove links");
                zeno::scope_exit scope([=]() { pModel->endTransaction(); });
                for (ZenoFullLink* link : seledLinks) {
                    auto linkIdx = link->linkInfo();
                    pModel->removeLink(linkIdx, true);
                }
            });

            menu->addAction(pTransferToNetLabel);
            menu->addAction(pDelete);

            menu->exec(QCursor::pos());
            menu->deleteLater();
            return true;
        }

        if (false && pNetLabel)
        {
            QMenu* menu = new QMenu;
            QAction* pDelete = new QAction(QObject::tr("Delete Net Label"));
            menu->addAction(pDelete);
            QObject::connect(pDelete, &QAction::triggered, [=]() {
                IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
                ZASSERT_EXIT(pModel);
                QString netlabel = pNetLabel->toPlainText();
                QModelIndex paramIdx = pNetLabel->paramIdx();
                pModel->removeNetLabel(subgIdx, paramIdx);
            });
            menu->exec(QCursor::pos());
            menu->deleteLater();
            return true;
        }

        if (pFocusItem)
        {
            //dispatch to item.
            return false;
        }

        NODE_CATES cates = zenoApp->graphsManagment()->currentModel()->getCates();
        auto m_menu = new ZenoNewnodeMenu(subgIdx, cates, scenePos);
        m_menu->setEditorFocus();
        m_menu->exec(QCursor::pos());
        m_menu->deleteLater();
        return true;
    }
    return false;
}