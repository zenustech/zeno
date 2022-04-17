#include "zenocheckbutton.h"


ZenoCheckButton::ZenoCheckButton(QWidget* parent)
	: _base(parent)
	, m_bChecked(false)
	, m_bHover(false)
{
	setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
}

ZenoCheckButton::ZenoCheckButton(const QIcon& icon, const QIcon& iconOn, QWidget* parent)
	: _base(parent)
	, m_icon(icon)
	, m_iconOn(iconOn)
	, m_bChecked(false)
	, m_bHover(false)
{
	setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
}

ZenoCheckButton::~ZenoCheckButton()
{
}

bool ZenoCheckButton::isChecked() const
{
	return m_bChecked;
}

void ZenoCheckButton::setChecked(bool bChecked)
{
	m_bChecked = bChecked;
	update();
}

void ZenoCheckButton::setIcons(const QIcon& icon, const QIcon& iconOn)
{
	m_icon = icon;
	m_iconOn = iconOn;
}

void ZenoCheckButton::setSize(const QSize& szIcon, const QMargins& margins)
{
	m_szIcon = szIcon;
	m_margins = margins;
}

QSize ZenoCheckButton::sizeHint() const
{
	return QSize(m_szIcon.width() + m_margins.left() + m_margins.right(),
		m_szIcon.height() + m_margins.top() + m_margins.bottom());
}

void ZenoCheckButton::enterEvent(QEvent* event)
{
	m_bHover = true;
	_base::enterEvent(event);
	update();
}

void ZenoCheckButton::leaveEvent(QEvent* event)
{
	m_bHover = false;
	_base::leaveEvent(event);
	update();
}

void ZenoCheckButton::mousePressEvent(QMouseEvent* event)
{
	_base::mousePressEvent(event);
}

void ZenoCheckButton::mouseReleaseEvent(QMouseEvent* event)
{
	_base::mouseReleaseEvent(event);
	bool bChecked = !m_bChecked;
	setChecked(bChecked);
	emit toggled(bChecked);
	update();
}

void ZenoCheckButton::paintEvent(QPaintEvent* event)
{
	QPainter painter(this);

	QIcon icon = m_icon;
	if (m_bChecked || m_bHover)
	{
		icon = m_iconOn;
	}

	QRect rc = rect();
	if (m_bChecked)
	{
		static int borderWidth = 2;
		painter.setPen(QPen(QBrush(Qt::white), borderWidth));
		int x = rc.left() + borderWidth / 2;
		painter.drawLine(QLine(QPoint(x, rc.top()), QPoint(x, rc.bottom())));
	}

	rc -= m_margins;
	icon.paint(&painter, rc);
}
