#include "zveceditor.h"
#include <zenoui/style/zenostyle.h>


ZVecEditor::ZVecEditor(const QVector<qreal>& vec, bool bFloat, int deflSize, QString styleCls, QWidget* parent)
	: QWidget(parent)
	, m_bFloat(bFloat)
{
	QHBoxLayout* pLayout = new QHBoxLayout;
	pLayout->setContentsMargins(0, 0, 0, 0);
	pLayout->setSpacing(5);
	int n = deflSize;
	if (!vec.isEmpty())
		n = vec.size();
	m_editors.resize(n);
	for (int i = 0; i < m_editors.size(); i++)
	{
		m_editors[i] = new QLineEdit;
        m_editors[i]->setFixedWidth(ZenoStyle::dpiScaled(64));
		m_editors[i]->setProperty("cssClass", styleCls);
		if (!vec.isEmpty())
			m_editors[i]->setText(QString::number(vec[i]));
		pLayout->addWidget(m_editors[i]);
		connect(m_editors[i], &QLineEdit::editingFinished, this, [=]() {
			emit editingFinished();
		});
	}
	setLayout(pLayout);
	setStyleSheet("ZVecEditor { background: transparent; } ");
}

QVector<qreal> ZVecEditor::vec() const
{
	QVector<qreal> v;
	for (int i = 0; i < m_editors.size(); i++)
	{
		v.append(m_editors[i]->text().toDouble());
	}
	return v;
}

void ZVecEditor::onValueChanged(const QVector<qreal>& vec)
{
	//todo: some vector without init is a empty vec, need to unify later.
	if (vec.size() != m_editors.size())
		return;
	for (int i = 0; i < m_editors.size(); i++)
	{
		m_editors[i]->setText(QString::number(vec[i]));
	}
}