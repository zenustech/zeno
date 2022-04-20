#ifndef __ZENO_CHECK_BUTTON_H__
#define __ZENO_CHECK_BUTTON_H__

#include <QtWidgets>

class ZenoCheckButton : public QWidget
{
	Q_OBJECT
	typedef QWidget _base;

public:
	ZenoCheckButton(QWidget* parent = nullptr);
	ZenoCheckButton(const QIcon& icon, const QIcon& iconOn, QWidget* parent = nullptr);
	~ZenoCheckButton();
	bool isChecked() const;
	void setChecked(bool bChecked);
	void setIcons(const QIcon& icon, const QIcon& iconOn);
	void setSize(const QSize& szIcon, const QMargins& margins);
	QSize sizeHint() const override;

protected:
	void enterEvent(QEvent* event) override;
	void leaveEvent(QEvent* event) override;
	void mousePressEvent(QMouseEvent* event) override;
	void mouseReleaseEvent(QMouseEvent* event) override;
	void paintEvent(QPaintEvent* event) override;

signals:
	void clicked();
	void toggled(bool);

private:
	QIcon m_icon, m_iconOn;	//svg
	QSize m_szIcon;
	QMargins m_margins;
	bool m_bHover;
	bool m_bChecked;
};

#endif
