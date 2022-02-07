#include "layerwidget.h"
#include "designermainwin.h"
#include "nodesview.h"
#include "styletabwidget.h"
#include "layertreeitemdelegate.h"
#include "nodeswidget.h"
#include "util.h"


NodesView* getCurrentView(QWidget* pWidget)
{
    QWidget* p = pWidget;
    while (p)
    {
        if (DesignerMainWin* pWin = qobject_cast<DesignerMainWin*>(p))
        {
            return pWin->getTabWidget()->getCurrentView();
        }
        p = p->parentWidget();
    }
    return nullptr;
}


LayerTreeView::LayerTreeView(QWidget* parent)
    : QTreeView(parent)
{
    setItemDelegate(new LayerTreeitemDelegate(this));
    setMinimumWidth(320);
}

QSize LayerTreeView::sizeHint() const
{
    QSize sz = QTreeView::sizeHint();
    if (model() == nullptr || model()->rowCount() == 0)
        return sz;

    int nToShow = model()->rowCount();
    return QSize(700, sz.height());
}

void LayerTreeView::mousePressEvent(QMouseEvent* e)
{
    QTreeView::mousePressEvent(e);
}

void LayerTreeView::mouseMoveEvent(QMouseEvent* e)
{
    QTreeView::mouseMoveEvent(e);
}

void LayerTreeView::updateHoverState(QPoint pos)
{
}

////////////////////////////////////////////////////////////
LayerWidget::LayerWidget(QWidget* parent)
    : QWidget(parent)
    , m_pLayer(nullptr)
{
    QVBoxLayout* pLayout = new QVBoxLayout;
    m_pLayer = new LayerTreeView;

    pLayout->addWidget(new QLabel(tr("Layer")));
    pLayout->addWidget(m_pLayer);

    setLayout(pLayout);
}

void LayerWidget::setModel(QStandardItemModel* model, QItemSelectionModel* selection)
{
    m_pLayer->setModel(model);
    if (selection) {
        m_pLayer->setSelectionModel(selection);
        m_pLayer->expandAll();
    }
}

void LayerWidget::resetModel()
{
    auto view = getCurrentView(this);

    NodesWidget* pTab = getMainWindow()->getCurrentTab();
    if (pTab) {
        QStandardItemModel *model = pTab->model();
        QItemSelectionModel *selection = pTab->selectionModel();
        m_pLayer->setModel(model);
        m_pLayer->setSelectionModel(selection);
        m_pLayer->expandAll();
    }
}