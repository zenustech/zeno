#ifndef __Z_EXPANDABLE_SECTION_H__
#define __Z_EXPANDABLE_SECTION_H__

#include <QtWidgets>

class ZIconLabel;

class ZExpandableSection : public QWidget
{
	Q_OBJECT
public:
	explicit ZExpandableSection(const QString& title, QWidget* parent = nullptr);
	void setContentLayout(QLayout* layout);

public slots:
	void toggle(bool collasped);

private:
	QGridLayout* m_mainLayout;
	ZIconLabel* m_collaspBtn;
	QScrollArea* m_contentArea;
	QWidget* m_contentWidget;
};

#endif