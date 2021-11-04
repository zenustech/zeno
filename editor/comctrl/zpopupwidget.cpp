#include "zpopupwidget.h"


ZPopupWidget::ZPopupWidget(QWidget* parent)
	: QWidget(parent)
	, m_pContentWidget(nullptr)
{
	setMouseTracking(true);
    setWindowFlag(Qt::Popup);
	m_layout = new QVBoxLayout;
	m_layout->setMargin(0);
	setLayout(m_layout);
}

ZPopupWidget::~ZPopupWidget()
{
	m_layout->removeWidget(m_pContentWidget);
}

void ZPopupWidget::setContentWidget(QWidget* contentWidget)
{
	m_pContentWidget = contentWidget;
	m_layout->addWidget(m_pContentWidget);
}

void ZPopupWidget::hideEvent(QHideEvent* event)
{
	emit aboutToHide();
}

void ZPopupWidget::closeEvent(QCloseEvent* event)
{
	QWidget::closeEvent(event);
	emit aboutToHide();
}

void ZPopupWidget::exec(int top, int left, int width, int height)
{
	if (!m_pContentWidget)
		return;

	QSize sz = sizeHint();
	if (m_pContentWidget && m_pContentWidget->testAttribute(Qt::WA_Resized))
	{
		if (m_pContentWidget->size().isValid())
		{
			sz = m_pContentWidget->size();
		}
	}
	setGeometry(top, left, sz.width(), sz.height());
	QEventLoop eventLoop;
	show();
	connect(this, SIGNAL(aboutToHide()), &eventLoop, SLOT(quit()));
	eventLoop.exec();
}