#include "layerwidget.h"


LayerTreeView::LayerTreeView(QWidget* parent)
    : QTreeView(parent)
{

}

LayerWidget::LayerWidget(QWidget* parent)
    : QWidget(parent)
{
    QVBoxLayout* pLayout = new QVBoxLayout;
    m_pHeader = new LayerTreeView;
    m_pBody = new LayerTreeView;

    pLayout->addWidget(new QLabel(tr("Layer")));
    pLayout->addWidget(m_pHeader);
    pLayout->addWidget(m_pBody);

    setLayout(pLayout);
}