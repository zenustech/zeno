#include "zveceditor.h"
#include <zenoui/style/zenostyle.h>
#include <zenomodel/include/uihelper.h>
#include "zlineedit.h"


ZVecEditor::ZVecEditor(const UI_VECTYPE& vec, bool bFloat, int deflSize, QString styleCls, QWidget* parent)
	: QWidget(parent)
	, m_bFloat(bFloat)
    , m_deflSize(deflSize)
    , m_styleCls(styleCls)
{
    initUI(vec);
}

void ZVecEditor::initUI(const UI_VECTYPE& vec)
{
    QHBoxLayout* pLayout = new QHBoxLayout;
    pLayout->setContentsMargins(0, 0, 0, 0);
    pLayout->setSpacing(5);
    int n = m_deflSize;
    if (!vec.isEmpty())
        n = vec.size();
    m_editors.resize(n);
    for (int i = 0; i < m_editors.size(); i++)
    {
        m_editors[i] = new ZLineEdit;
        m_editors[i]->setNumSlider(UiHelper::getSlideStep("", m_bFloat ? CONTROL_FLOAT : CONTROL_INT));
        //m_editors[i]->setFixedWidth(ZenoStyle::dpiScaled(64));
        m_editors[i]->setProperty("cssClass", m_styleCls);
        if (!vec.isEmpty())
            m_editors[i]->setText(QString::number(vec[i]));
        pLayout->addWidget(m_editors[i]);
        connect(m_editors[i], &ZLineEdit::editingFinished, this, [=]() {
            emit editingFinished();
            });
    }
    setLayout(pLayout);
    setStyleSheet("ZVecEditor { background: transparent; } ");
}

bool ZVecEditor::isFloat() const
{
    return m_bFloat;
}

UI_VECTYPE ZVecEditor::vec() const
{
	UI_VECTYPE v;
	for (int i = 0; i < m_editors.size(); i++)
	{
        if (m_bFloat)
		    v.append(m_editors[i]->text().toFloat());
        else
            v.append(m_editors[i]->text().toInt());
	}
	return v;
}

void ZVecEditor::setVec(const UI_VECTYPE& vec, bool bFloat)
{
    if (bFloat != m_bFloat || vec.size() != m_editors.size())
    {
        initUI(vec);
    }
    else
    {
        for (int i = 0; i < m_editors.size(); i++)
        {
            m_editors[i]->setText(QString::number(vec[i]));
        }
    }
}