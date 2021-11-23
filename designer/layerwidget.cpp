#include "layerwidget.h"
#include "designermainwin.h"
#include "nodesview.h"
#include "styletabwidget.h"

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

}

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

void LayerWidget::resetModel()
{
    auto view = getCurrentView(this);
    QStandardItemModel* model = view->findChild<QStandardItemModel*>(NODE_MODEL_NAME);
    QItemSelectionModel* selection = view->findChild<QItemSelectionModel*>(NODE_SELECTION_MODEL);
    m_pLayer->setModel(model);
    m_pLayer->setSelectionModel(selection);
    m_pLayer->expandAll();
}