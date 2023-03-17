#include "zexpandablesection.h"
#include "zlabel.h"
#include "../style/zenostyle.h"
#include "zlinewidget.h"


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

QSize ZScrollArea::minimumSizeHint() const
{
    if (QWidget* pWidget = this->widget())
    {
        int cnt = pWidget->layout()->count();
        if (cnt == 0)
        {
            return QSize(0, 0);
        }
    }
    return QScrollArea::minimumSizeHint();
}

QSize ZScrollArea::sizeHint() const
{
    //mock QScrollArea::sizeHint()

    int f = 2 * frameWidth();
    QSize sz(f, f);
    int h = fontMetrics().height();
    if (QWidget* pWidget = this->widget())
    {
        int cnt = pWidget->layout()->count();
        if (cnt > 0)
        {
            widgetSize = widgetResizable() ? pWidget->sizeHint() : pWidget->size();
            sz += widgetSize;
        }
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
    , m_collaspBtn(nullptr)
{
	m_contentArea = new ZScrollArea(this);
	m_mainLayout = new QVBoxLayout(this);

	m_contentArea->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	m_contentArea->setMinimumHeight(0);
	m_contentArea->setFrameShape(QFrame::NoFrame);
	m_contentArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	m_contentArea->setWidgetResizable(true);

	m_mainLayout->setSpacing(0);
	m_mainLayout->setContentsMargins(0, 0, 0, 0);

	m_mainLayout->addWidget(initTitleWidget(title));
	m_mainLayout->addWidget(m_contentArea);

	setLayout(m_mainLayout);

	setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Maximum);
}

ZExpandableSection::~ZExpandableSection()
{
}

void ZExpandableSection::setContentLayout(QLayout* contentLayout)
{
    ZContentWidget* contentWidget = new ZContentWidget;
    contentWidget->setLayout(contentLayout);
    contentWidget->setAutoFillBackground(true);
    contentWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    QPalette pal = this->palette();
    pal.setColor(QPalette::Window, QColor(45, 50, 58));
    contentWidget->setPalette(pal);

    m_contentArea->setWidget(contentWidget);
    update();
}

void ZExpandableSection::updateGeo()
{
    m_contentArea->updateGeometry();
    updateGeometry();
}

QWidget* ZExpandableSection::initTitleWidget(const QString& title)
{
	QWidget* pWidget = new QWidget;

	QVBoxLayout* pLayout = new QVBoxLayout;

	//pLayout->addWidget(new ZPlainLine(1, QColor(0, 0, 0)));
	pLayout->setContentsMargins(0, 0, 0, 0);
	pLayout->setSpacing(0);

	QHBoxLayout* titleLayout = new QHBoxLayout;

	QLabel* plblTitle = new QLabel(title);
	plblTitle->setProperty("cssClass", "proppanel-sectionname");

    m_collaspBtn = new ZIconLabel;
    m_collaspBtn->setIcons(ZenoStyle::dpiScaledSize(QSize(24, 24)), ":/icons/ic_parameter_fold.svg", "", ":/icons/ic_parameter_unfold.svg");
    m_collaspBtn->toggle();

	titleLayout->addWidget(m_collaspBtn);
	titleLayout->addWidget(plblTitle);
	titleLayout->setContentsMargins(0, 0, 0, 0);

	pLayout->addLayout(titleLayout);

	pWidget->setLayout(pLayout);

	pWidget->setAutoFillBackground(true);
	QPalette pal = this->palette();
	pal.setColor(QPalette::Window, QColor(60, 66, 78));
	pWidget->setPalette(pal);
	pWidget->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);

	connect(m_collaspBtn, &ZIconLabel::toggled, this, &ZExpandableSection::toggle);

	return pWidget;
}

QLayout* ZExpandableSection::contentLayout() const
{
	QWidget* pContentWid = m_contentArea->widget();
	if (!pContentWid) return nullptr;
	return pContentWid->layout();
}

void ZExpandableSection::toggle(bool)
{
    bool bCollasped = m_contentArea->isVisible();
    m_contentArea->setVisible(!bCollasped);
    updateGeometry();
    emit stateChanged(bCollasped);
}

void ZExpandableSection::setCollasped(bool bOn)
{
    m_collaspBtn->toggle(!bOn);
    m_contentArea->setVisible(!bOn);
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

QString ZExpandableSection::title() const
{
    return m_title;
}

void ZExpandableSection::mousePressEvent(QMouseEvent* event)
{
    //hit test.
    QWidget::mousePressEvent(event);
}
