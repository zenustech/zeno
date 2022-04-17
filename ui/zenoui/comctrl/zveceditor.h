#ifndef __ZVEC_EDITOR_H__
#define __ZVEC_EDITOR_H__

#include <QtWidgets>

class ZVecEditor : public QWidget
{
	Q_OBJECT
public:
	ZVecEditor(const QVector<qreal>& vec, bool bFloat, int deflSize, QString styleCls, QWidget* parent = nullptr);
	QVector<qreal> vec() const;

signals:
	void valueChanged(QVector<qreal>);
	void editingFinished();

public slots:
	void onValueChanged(const QVector<qreal>&);

private:
	QVector<QLineEdit*> m_editors;
	bool m_bFloat;
};


#endif