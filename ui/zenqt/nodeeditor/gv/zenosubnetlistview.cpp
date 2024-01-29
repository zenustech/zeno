#include "zenosubnetlistview.h"
#include "zenoapplication.h"
#include "style/zenostyle.h"
#include "zsubnetlistitemdelegate.h"
#include "widgets/zlabel.h"
#include "uicommon.h"
#include "model/assetsmodel.h"
#include <zeno/utils/logger.h>
#include "util/log.h"


ZenoSubnetListView::ZenoSubnetListView(QWidget* parent)
    : QListView(parent)
{
    setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Expanding);
    setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    setFrameShape(QFrame::NoFrame);
    setFrameShadow(QFrame::Plain);
}

ZenoSubnetListView::~ZenoSubnetListView()
{
}

void ZenoSubnetListView::initModel(AssetsModel* pModel)
{
    setModel(pModel);
    //setItemDelegate(new ZSubnetListItemDelegate(pModel, this));
    viewport()->setAutoFillBackground(false);
    update();
}

void ZenoSubnetListView::edittingNew()
{
    AssetsModel* pModel = qobject_cast<AssetsModel*>(model());
    ZASSERT_EXIT(pModel);

    //SubGraphModel* pSubModel = new SubGraphModel(pModel);
    //pModel->appendSubGraph(pSubModel);

    //const QModelIndex& idx = pModel->indexBySubModel(pSubModel);
    //setCurrentIndex(idx);
    //edit(idx);
}

void ZenoSubnetListView::closeEditor(QWidget* editor, QAbstractItemDelegate::EndEditHint hint)
{
    QModelIndex idx = currentIndex();
    QListView::closeEditor(editor, hint);
    
    switch (hint)
    {
        case QAbstractItemDelegate::RevertModelCache:
        {
            //pModel->revert(idx);
            break;
        }
        case QAbstractItemDelegate::SubmitModelCache:
        {
            //activate the tab widget.
            QString subgName = idx.data().toString();
            emit graphToBeActivated(subgName);
            break;
        }
    }
}

QSize ZenoSubnetListView::sizeHint() const
{
    if (model() == nullptr)
        return QListView::sizeHint();

    if (model()->rowCount() == 0)
        return QListView::sizeHint();

    int nToShow = model()->rowCount();
    return QSize(sizeHintForColumn(0), nToShow * sizeHintForRow(0));
}

void ZenoSubnetListView::paintEvent(QPaintEvent* e)
{
    QListView::paintEvent(e);
}
