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
	virtual QSize minimumSizeHint() const override;

private:
    mutable QSize widgetSize;
};


class ZExpandableSection : public QWidget
{
	Q_OBJECT
public:
	explicit ZExpandableSection(const QString& title, QWidget* parent = nullptr);
	~ZExpandableSection();
	void setContentLayout(QLayout* layout);
	QLayout* contentLayout() const;
	virtual QSize sizeHint() const override;
    virtual QSize minimumSizeHint() const override;
	QString title() const;

protected:
    void mousePressEvent(QMouseEvent* event) override;

signals:
	void stateChanged(bool);

public slots:
	void toggle(bool collasped);
	void setCollasped(bool bOn);
	void updateGeo();

private:
	QWidget* initTitleWidget(const QString& title/* other ui params*/);
	QString m_title;

	QVBoxLayout* m_mainLayout;
	ZScrollArea* m_contentArea;
	QWidget* m_contentWidget;
	ZIconLabel* m_collaspBtn;
};

#endif