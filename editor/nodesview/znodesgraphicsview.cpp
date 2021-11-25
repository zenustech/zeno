#include "znodesgraphicsview.h"


ZNodesGraphicsView::ZNodesGraphicsView(QWidget* parent)
    : QWidget(parent)
    , m_view(nullptr)
    , m_scene(nullptr)
{
    m_scene = new QDMGraphicsScene;
	m_view = new QDMGraphicsView;
	m_view->setScene(m_scene);
	m_view->setBackgroundBrush(QBrush(QColor(38, 50, 56), Qt::SolidPattern));
	m_view->setFrameShape(QFrame::NoFrame);

    QVBoxLayout* pLayout = new QVBoxLayout;
    pLayout->setMargin(0);
    pLayout->addWidget(m_view);

    setLayout(pLayout);

    initNodes();
}

void ZNodesGraphicsView::initNodes()
{
    m_scene->addNode(new QDMGraphicsNode);
}