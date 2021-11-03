#include "zmenubutton.h"

#include "../style/zenostyle.h"
#include "../style/zstyleoption.h"


ZMenuButton::ZMenuButton(ButtonOption option, const QIcon& icon, const QSize& iconSize, const QString& text, QWidget* parent)
	: ZToolButton(option, icon, iconSize, text, parent)
	, func_createContentWid(nullptr)
{
	setMouseTracking(true);
	connect(this, SIGNAL(clicked()), this, SIGNAL(popup()));
	connect(this, SIGNAL(popup()), this, SLOT(popupChildWidget()));
}

ZMenuButton::~ZMenuButton()
{
}

void ZMenuButton::setCreateContentCallback(std::function<QWidget* ()> func)
{
	func_createContentWid = func;
}

void ZMenuButton::popupChildWidget()
{
	if (func_createContentWid)
	{
		ZPopupWidget popup(this);

		connect(this, SIGNAL(popout()), &popup, SIGNAL(aboutToHide()));

		QWidget* pContentWidget = func_createContentWid();
		popup.setContentWidget(pContentWidget);

		QPoint pGlobal = mapToGlobal(QPoint(0, 0));
		const int margin = 5;
		setDown(true);

		int nWidth = 300; pContentWidget->width();
		int nHeight = pContentWidget->height();

		popup.exec(pGlobal.x(), pGlobal.y() + height() + margin, nWidth, nHeight);
		setDown(false);
	}
}

void ZMenuButton::initStyleOption(ZStyleOptionToolButton* option) const
{
	ZToolButton::initStyleOption(option);
	option->features |= QStyleOptionToolButton::Menu;
}

void ZMenuButton::paintEvent(QPaintEvent* event)
{
	ZToolButton::paintEvent(event);
}

bool ZMenuButton::event(QEvent* e)
{
	return ZToolButton::event(e);
}