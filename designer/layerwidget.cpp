#include "layerwidget.h"


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

void LayerWidget::setModel(QStandardItemModel* model, QItemSelectionModel* selectionModel)
{
    m_pLayer->setModel(model);
    m_pLayer->setSelectionModel(selectionModel);
    m_pLayer->expandAll();
}