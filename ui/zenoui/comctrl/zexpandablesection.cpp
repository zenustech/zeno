#include "zexpandablesection.h"
#include "zlabel.h"
#include "../style/zenostyle.h"


ZExpandableSection::ZExpandableSection(const QString& title, QWidget* parent)
	: QWidget(parent)
	, m_mainLayout(nullptr)
	, m_animation(nullptr)
	, m_contentArea(nullptr)
{
	m_animation = new QParallelAnimationGroup(this);
	m_contentArea = new QScrollArea(this);
	m_mainLayout = new QGridLayout(this);

	QLabel* plblTitle = new QLabel(title);
	plblTitle->setProperty("cssClass", "proppanel-sectionname");

	m_collaspBtn = new ZIconLabel;
	m_collaspBtn->setIcons(ZenoStyle::dpiScaledSize(QSize(24, 24)), ":/icons/ic_parameter_fold.svg", "", ":/icons/ic_parameter_unfold.svg");

	m_contentArea->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
	m_contentArea->setMaximumHeight(0);
	m_contentArea->setMinimumHeight(0);
	m_contentArea->setStyleSheet("QScrollArea {background: transparent;}");
	m_contentArea->setFrameShape(QFrame::NoFrame);

	m_animation->addAnimation(new QPropertyAnimation(this, "minimumHeight"));
	m_animation->addAnimation(new QPropertyAnimation(this, "maximumHeight"));
	m_animation->addAnimation(new QPropertyAnimation(m_contentArea, "maximumHeight"));

	m_mainLayout->setVerticalSpacing(0);
	m_mainLayout->setContentsMargins(15, 15, 15, 15);

	int row = 0;
	m_mainLayout->addWidget(m_collaspBtn, 0, 0);
	m_mainLayout->addWidget(plblTitle, 0, 1);
	m_mainLayout->addWidget(m_contentArea, 1, 1);

	setLayout(m_mainLayout);

	connect(m_collaspBtn, &ZIconLabel::toggled, this, &ZExpandableSection::toggle);
}

void ZExpandableSection::setContentLayout(QLayout* contentLayout)
{
	delete m_contentArea->layout();
	m_contentArea->setLayout(contentLayout);

	const auto collapsedHeight = sizeHint().height() - m_contentArea->maximumHeight();
	auto contentHeight = contentLayout->sizeHint().height();

	for (int i = 0; i < m_animation->animationCount() - 1; i++)
	{
		QPropertyAnimation* SectionAnimation = static_cast<QPropertyAnimation*>(m_animation->animationAt(i));
		SectionAnimation->setDuration(m_duration);
		SectionAnimation->setStartValue(collapsedHeight);
		SectionAnimation->setEndValue(collapsedHeight + contentHeight);
	}

	QPropertyAnimation* contentAnimation = static_cast<QPropertyAnimation*>(m_animation->animationAt(
		m_animation->animationCount() - 1));
	contentAnimation->setDuration(m_duration);
	contentAnimation->setStartValue(0);
	contentAnimation->setEndValue(contentHeight);

	//expand when inited.
	m_collaspBtn->toggle();
	m_contentArea->setMaximumHeight(contentHeight);
	m_animation->setDirection(QAbstractAnimation::Backward);
}

void ZExpandableSection::toggle(bool collasped)
{
	m_animation->setDirection(collasped ? QAbstractAnimation::Forward : QAbstractAnimation::Backward);
	m_animation->start();
}