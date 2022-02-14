#include "zenosubnettreeview.h"
#include "subnettreeitemdelegate.h"
#include <zenoui/model/graphsmodel.h>
#include <model/graphstreemodel.h>
#include <zenoui/include/igraphsmodel.h>


ZenoSubnetTreeView::ZenoSubnetTreeView(QWidget* parent)
    : QTreeView(parent)
{
    header()->setVisible(false);
	setFrameShape(QFrame::NoFrame);
	setFrameShadow(QFrame::Plain);
	setFocusPolicy(Qt::NoFocus);
	setAlternatingRowColors(false);
	initStyle();
}

void ZenoSubnetTreeView::initStyle()
{
	const QString treeViewStyleSheet =
		"\
			QTreeView\
		    {\
				background-color: rgb(43,43,43);\
				show-decoration-selected: 1;\
				font: 10pt 'HarmonyOS Sans';\
		    }\
\
			QTreeView::item {\
				color: #858280;\
				border: 1px solid transparent;\
			}\
\
			QTreeView::item:hover {\
				background-color: transparent;\
			}\
\
			QTreeView::item:selected {\
				border: 1px solid #4B9EF4;\
				background: #334960;\
				color: #ffffff;\
			}\
\
			QTreeView::item:selected:active{\
				background: #334960;\
				color: #ffffff;\
			}\
\
			QTreeView::item:selected:!active {\
				background: #334960;\
				color: #ffffff;\
			}\
\
		";
	this->setStyleSheet(treeViewStyleSheet);
}

ZenoSubnetTreeView::~ZenoSubnetTreeView()
{
}

void ZenoSubnetTreeView::initModel(IGraphsModel* pModel)
{
	GraphsModel* pPlainModel = qobject_cast<GraphsModel*>(pModel);
	Q_ASSERT(pPlainModel);
    GraphsTreeModel* pTreeModel = new GraphsTreeModel(pPlainModel, this);    //todo: put in managment.
    pTreeModel->init(pPlainModel);
    setModel(pTreeModel);
}

void ZenoSubnetTreeView::paintEvent(QPaintEvent* e)
{
    QTreeView::paintEvent(e);
}
