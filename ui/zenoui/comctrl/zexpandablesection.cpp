#include "zexpandablesection.h"
#include "zlabel.h"
#include "../style/zenostyle.h"


ZContentWidget::ZContentWidget(QWidget *parent) 
	: QWidget(parent)
{
}

QSize ZContentWidget::sizeHint() const
{
    QSize sz = QWidget::sizeHint();
    return sz;
}

QSize ZContentWidget::minimumSizeHint() const
{
    QSize sz = QWidget::minimumSizeHint();
    return sz;
}


ZScrollArea::ZScrollArea(QWidget* parent)
	: QScrollArea(parent)
{
}

QSize ZScrollArea::sizeHint() const
{
    //mock QScrollArea::sizeHint()
    int f = 2 * frameWidth();
    QSize sz(f, f);
    int h = fontMetrics().height();
    if (QWidget* pWidget = this->widget()) {
        if (!widgetSize.isValid())
            widgetSize = widgetResizable() ? pWidget->sizeHint() : pWidget->size();
        sz += widgetSize;
    }
    else {
        sz += QSize(12 * h, 8 * h);
    }
    if (verticalScrollBarPolicy() == Qt::ScrollBarAlwaysOn)
        sz.setWidth(sz.width() + verticalScrollBar()->sizeHint().width());
    if (horizontalScrollBarPolicy() == Qt::ScrollBarAlwaysOn)
        sz.setHeight(sz.height() + horizontalScrollBar()->sizeHint().height());
    if (sz.isValid())
        return sz;      //custom: return real widget size without bound.
    return sz.boundedTo(QSize(36 * h, 24 * h));
}



ZExpandableSection::ZExpandableSection(const QString& title, QWidget* parent)
	: QWidget(parent)
	, m_mainLayout(nullptr)
	, m_contentArea(nullptr)
	, m_contentWidget(nullptr)
	, m_title(title)
{
	m_contentArea = new ZScrollArea(this);
	m_mainLayout = new QGridLayout(this);

	QLabel* plblTitle = new QLabel(title);
	plblTitle->setProperty("cssClass", "proppanel-sectionname");

	m_collaspBtn = new ZIconLabel;
	m_collaspBtn->setIcons(ZenoStyle::dpiScaledSize(QSize(24, 24)), ":/icons/ic_parameter_fold.svg", "", ":/icons/ic_parameter_unfold.svg");
	m_collaspBtn->toggle();

	m_contentArea->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	m_contentArea->setMinimumHeight(0);
	m_contentArea->setProperty("cssClass", "proppanel");
	m_contentArea->setFrameShape(QFrame::NoFrame);
	m_contentArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	m_contentArea->setWidgetResizable(true);

	m_mainLayout->setVerticalSpacing(0);
	m_mainLayout->setContentsMargins(15, 15, 15, 15);

	int row = 0;
	m_mainLayout->addWidget(m_collaspBtn, 0, 0);
	m_mainLayout->addWidget(plblTitle, 0, 1);
	m_mainLayout->addWidget(m_contentArea, 1, 1);

	setLayout(m_mainLayout);

	connect(m_collaspBtn, &ZIconLabel::toggled, this, &ZExpandableSection::toggle);
	setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Expanding);
}

void ZExpandableSection::setContentLayout(QLayout* contentLayout)
{
    ZContentWidget* contentWidget = new ZContentWidget;
    contentWidget->setLayout(contentLayout);
    contentWidget->setAutoFillBackground(true);
	contentWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    QPalette pal = this->palette();
    pal.setColor(QPalette::Window, QColor(37, 37, 37));
    contentWidget->setPalette(pal);

    m_contentArea->setWidget(contentWidget);
	update();
}

void ZExpandableSection::toggle(bool collasped)
{
    m_contentArea->setVisible(!m_contentArea->isVisible());
    update();
}

QSize ZExpandableSection::sizeHint() const
{
    QSize sz = QWidget::sizeHint();
    return sz;
}

QSize ZExpandableSection::minimumSizeHint() const
{
    QSize sz = QWidget::minimumSizeHint();
    return sz;
}

void ZExpandableSection::mousePressEvent(QMouseEvent* event)
{
	//hit test.
	QWidget::mousePressEvent(event);
}
