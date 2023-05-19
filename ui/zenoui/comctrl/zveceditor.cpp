#include "zveceditor.h"
#include <zenoui/style/zenostyle.h>
#include <zenomodel/include/uihelper.h>
#include "zlineedit.h"

ZVecEditor::ZVecEditor(const QVariant &vec, bool bFloat, int deflSize, QString styleCls, QWidget *parent)
	: QWidget(parent)
	, m_bFloat(bFloat)
    , m_deflSize(deflSize)
    , m_styleCls(styleCls)
{
    initUI(vec);
}

void ZVecEditor::initUI(const QVariant& vec)
{
    QHBoxLayout* pLayout = new QHBoxLayout;
    pLayout->setContentsMargins(0, 0, 0, 0);
    pLayout->setSpacing(5);
    int n = m_deflSize;
    if (vec.canConvert<UI_VECTYPE>())
        n = vec.value<UI_VECTYPE>().size();
    else
        n = vec.value<CURVES_DATA>().size();

    m_editors.resize(n);
    for (int i = 0; i < m_editors.size(); i++)
    {
        if (m_bFloat)
            m_editors[i] = new ZFloatLineEdit;
        else
            m_editors[i] = new ZLineEdit;

        m_editors[i]->setNumSlider(UiHelper::getSlideStep("", m_bFloat ? CONTROL_FLOAT : CONTROL_INT));
        //m_editors[i]->setFixedWidth(ZenoStyle::dpiScaled(64));
        m_editors[i]->setProperty("cssClass", m_styleCls);
        if (vec.canConvert<UI_VECTYPE>())
            setText(vec.value<UI_VECTYPE>().at(i), m_editors[i]);
        else if (vec.canConvert<CURVES_DATA>()) {
            CURVES_DATA curves = vec.value<CURVES_DATA>();
            QString key = UiHelper::getCurveKey(i);
            if (curves.contains(key)) {
                m_editors[i]->setProperty(g_keyFrame, QVariant::fromValue(curves[key]));
            }
        }

        pLayout->addWidget(m_editors[i]);
        connect(m_editors[i], &ZLineEdit::editingFinished, this, &ZVecEditor::editingFinished);
    }
    setLayout(pLayout);
    setStyleSheet("ZVecEditor { background: transparent; } ");
}

bool ZVecEditor::isFloat() const
{
    return m_bFloat;
}

QVariant ZVecEditor::vec() const
{
	QVariant value;
    CURVES_DATA datas;
    UI_VECTYPE vec;
    bool bKeyFrame = false;
	for (int i = 0; i < m_editors.size(); i++)
	{
        if (m_bFloat) {
            if (m_editors[i]->property(g_keyFrame).canConvert<CURVE_DATA>()) 
            {
                bKeyFrame = true;
                CURVE_DATA data = m_editors[i]->property(g_keyFrame).value<CURVE_DATA>();
                QString key = UiHelper::getCurveKey(i);
                if (data.key != key) {
                    data.key = key;
                    m_editors[i]->setProperty(g_keyFrame, QVariant::fromValue(data));
                }
                datas.insert(data.key, data);
            } 
            else 
            {
                vec.append(m_editors[i]->text().toFloat());
            }
        }
        else
            vec.append(m_editors[i]->text().toInt());
	}
    if (!bKeyFrame) 
    {
        value = QVariant::fromValue(vec);
    }
    else 
    {
        if (datas.size() != m_editors.size())
        {
            for (int i = 0; i < m_editors.size(); i++) {
                QString key = UiHelper::getCurveKey(i);
                if (datas.contains(key))
                    continue;
                CURVE_DATA data;
                if (ZFloatLineEdit *lineEdit = qobject_cast<ZFloatLineEdit *>(m_editors[i])) {
                    lineEdit->getDelfCurveData(data, false);
                }
                data.key = key;
                datas.insert(data.key, data);
            }
        }
        value = QVariant::fromValue(datas);
    }
    return value;
}

void ZVecEditor::setVec(const QVariant& vec, bool bFloat)
{
    int size = m_editors.size();
    if (vec.canConvert<UI_VECTYPE>())
        size = vec.value<UI_VECTYPE>().size();
    else
        size = vec.value<CURVES_DATA>().size();
    if (bFloat != m_bFloat || size != m_editors.size())
    {
        initUI(vec);
    }
    else
    {
        for (int i = 0; i < m_editors.size(); i++) 
        {
            if (vec.canConvert<UI_VECTYPE>())
                setText(vec.value<UI_VECTYPE>().at(i), m_editors[i]);
            else if (vec.canConvert<CURVES_DATA>()) {
                CURVES_DATA curves = vec.value<CURVES_DATA>();
                QString key = UiHelper::getCurveKey(i);
                if (curves.contains(key)) 
                {
                    m_editors[i]->setProperty(g_keyFrame, QVariant::fromValue(curves[key]));
                }
            }
        }
    }
}

void ZVecEditor::setText(const QVariant &value, ZLineEdit* lineEdit) 
{
    if (m_bFloat) {
        lineEdit->setText(QString::number(value.toFloat()));
    } else {
        lineEdit->setText(QString::number(value.toInt()));
    }
}