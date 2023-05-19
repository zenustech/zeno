#ifndef __ZVEC_EDITOR_H__
#define __ZVEC_EDITOR_H__

#include <QtWidgets>

class ZLineEdit;

class ZVecEditor : public QWidget
{
	Q_OBJECT
public:
	ZVecEditor(const QVariant& vec, bool bFloat, int deflSize, QString styleCls, QWidget* parent = nullptr);
	QVariant vec() const;
	bool isFloat() const;

signals:
    void valueChanged(QVariant);
	void editingFinished();

public slots:
	void setVec(const QVariant& vec, bool bFloat);

private:
	void initUI(const QVariant& vec);
    void setText(const QVariant &value, ZLineEdit*);

	QVector<ZLineEdit*> m_editors;
	int m_deflSize;
	QString m_styleCls;
	bool m_bFloat;
};


#endif