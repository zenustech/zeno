#ifndef __POPUP_WIDGET_H__
#define __POPUP_WIDGET_H__

#include <QtWidgets>

class ZPopupWidget : public QWidget
{
	Q_OBJECT
public:
	ZPopupWidget(QWidget* parent = nullptr);
	~ZPopupWidget();

	void setContentWidget(QWidget* contentWidget);
	void exec(int top, int left, int width, int height);

protected:
	void hideEvent(QHideEvent* event) override;
	void closeEvent(QCloseEvent* event) override;

signals:
	void aboutToHide();

private:
	QWidget* m_pContentWidget;
	QVBoxLayout* m_layout;
};

#endif
