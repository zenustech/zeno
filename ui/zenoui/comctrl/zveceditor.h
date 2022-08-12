#ifndef __ZVEC_EDITOR_H__
#define __ZVEC_EDITOR_H__

#include <QtWidgets>
#include "../model/modeldata.h"

class ZVecEditor : public QWidget
{
	Q_OBJECT
public:
	ZVecEditor(const UI_VECTYPE& vec, bool bFloat, int deflSize, QString styleCls, QWidget* parent = nullptr);
	UI_VECTYPE vec() const;

signals:
	void valueChanged(UI_VECTYPE);
	void editingFinished();

public slots:
	void onValueChanged(const UI_VECTYPE&);

private:
	QVector<QLineEdit*> m_editors;
	bool m_bFloat;
};


#endif