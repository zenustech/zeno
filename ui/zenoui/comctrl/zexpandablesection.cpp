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



ZExpandableSection::ZExpandableSection(const QString& title, QWidget* parent)
	: QWidget(parent)
	, m_mainLayout(nullptr)
	, m_contentArea(nullptr)
	, m_contentWidget(nullptr)
{
	m_contentArea = new QScrollArea(this);
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

	setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Maximum);
}

void ZExpandableSection::setContentLayout(QLayout* contentLayout)
{
    ZContentWidget* contentWidget = new ZContentWidget;
    contentWidget->setLayout(contentLayout);
    contentWidget->setAutoFillBackground(true);
	contentWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    QPalette pal = this->palette();
    pal.setColor(QPalette::Window, QColor(44, 51, 58));
    contentWidget->setPalette(pal);

    m_contentArea->setWidget(contentWidget);
	update();
}

QWidget* ZExpandableSection::initTitleWidget(const QString& title)
{
	QWidget* pWidget = new QWidget;

	QVBoxLayout* pLayout = new QVBoxLayout;

	pLayout->addWidget(new ZPlainLine(1, QColor(0, 0, 0)));
	pLayout->setContentsMargins(0, 0, 0, 0);
	pLayout->setSpacing(0);

	QHBoxLayout* titleLayout = new QHBoxLayout;

	QLabel* plblTitle = new QLabel(title);
	plblTitle->setProperty("cssClass", "proppanel-sectionname");

	ZIconLabel* collaspBtn = new ZIconLabel;
	collaspBtn->setIcons(ZenoStyle::dpiScaledSize(QSize(24, 24)), ":/icons/ic_parameter_fold.svg", "", ":/icons/ic_parameter_unfold.svg");
	collaspBtn->toggle();

	titleLayout->addWidget(collaspBtn);
	titleLayout->addWidget(plblTitle);
	titleLayout->setContentsMargins(0, 0, 0, 0);

	pLayout->addLayout(titleLayout);

	pWidget->setLayout(pLayout);

	pWidget->setAutoFillBackground(true);
	QPalette pal = this->palette();
	pal.setColor(QPalette::Window, QColor(60, 66, 78));
	pWidget->setPalette(pal);

	connect(collaspBtn, &ZIconLabel::toggled, this, &ZExpandableSection::toggle);

	return pWidget;
}

QLayout* ZExpandableSection::contentLayout() const
{
	QWidget* pContentWid = m_contentArea->widget();
	if (!pContentWid) return nullptr;
	return pContentWid->layout();
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