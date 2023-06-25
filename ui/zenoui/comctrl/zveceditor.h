#ifndef __ZVEC_EDITOR_H__
#define __ZVEC_EDITOR_H__

#include <QtWidgets>
#include <zenomodel/include/modeldata.h>

class ZLineEdit;


class ZVecEditor : public QWidget
{
	Q_OBJECT
public:
	ZVecEditor(const QVariant& vec, bool bFloat, int deflSize, QString styleCls, QWidget* parent = nullptr);
	QVariant vec() const;
	bool isFloat() const;
    UI_VECTYPE text() const;
    int getCurrentEditor();
    void updateProperties(const QVector<QString>& properties);

signals:
    void valueChanged(QVariant);
	void editingFinished();

public slots:
	void setVec(const QVariant& vec, bool bFloat);

protected:
    bool eventFilter(QObject *watched, QEvent *event);

private:
	void initUI(const QVariant& vec);
    void setText(const QVariant &value, ZLineEdit*);

	QVector<ZLineEdit*> m_editors;
	int m_deflSize;
	QString m_styleCls;
	bool m_bFloat;
};


#endif