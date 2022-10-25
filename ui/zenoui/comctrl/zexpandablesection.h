#ifndef __Z_EXPANDABLE_SECTION_H__
#define __Z_EXPANDABLE_SECTION_H__

#include <QtWidgets>

class ZIconLabel;

class ZContentWidget : public QWidget
{
	Q_OBJECT
public:
    ZContentWidget(QWidget* parent = nullptr);
	virtual QSize sizeHint() const override;
    virtual QSize minimumSizeHint() const override;
};

class ZScrollArea : public QScrollArea
{
	Q_OBJECT
public:
	ZScrollArea(QWidget* parent = nullptr);
	virtual QSize sizeHint() const override;

private:
	mutable QSize widgetSize;
};

class ZExpandableSection : public QWidget
{
	Q_OBJECT
public:
	explicit ZExpandableSection(const QString& title, QWidget* parent = nullptr);
	void setContentLayout(QLayout* layout);
	virtual QSize sizeHint() const override;
    virtual QSize minimumSizeHint() const override;

protected:
	void mousePressEvent(QMouseEvent* event) override;

public slots:
	void toggle(bool collasped);

private:
	QString m_title;
	QGridLayout* m_mainLayout;
	ZIconLabel* m_collaspBtn;
	ZScrollArea* m_contentArea;
	QWidget* m_contentWidget;
};

#endif