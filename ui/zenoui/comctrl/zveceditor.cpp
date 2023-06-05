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

bool ZVecEditor::eventFilter(QObject *watched, QEvent *event) {
    if (event->type() == QEvent::ContextMenu) {
        for (int i = 0; i < m_editors.size(); i++) {
            if (m_editors[i] == watched) {
                qApp->sendEvent(this, event);
                return true;
            }
        }
    }
    return QWidget::eventFilter(watched, event);
}

void ZVecEditor::initUI(const QVariant &vec) {
    QHBoxLayout* pLayout = new QHBoxLayout;
    pLayout->setContentsMargins(0, 0, 0, 0);
    pLayout->setSpacing(5);
    int n = m_deflSize;
    if (vec.canConvert<UI_VECTYPE>())
        n = vec.value<UI_VECTYPE>().size();
    else if (vec.canConvert<UI_VECSTRING>())
        n = vec.value<UI_VECSTRING>().size();
    else
        n = vec.value<CURVES_DATA>().size();

    m_editors.resize(n);
    for (int i = 0; i < m_editors.size(); i++)
    {
        if (m_bFloat) {
            m_editors[i] = new ZFloatLineEdit;
            m_editors[i]->installEventFilter(this);
        }
        else
            m_editors[i] = new ZLineEdit;

        m_editors[i]->setNumSlider(UiHelper::getSlideStep("", m_bFloat ? CONTROL_FLOAT : CONTROL_INT));
        //m_editors[i]->setFixedWidth(ZenoStyle::dpiScaled(64));
        m_editors[i]->setProperty("cssClass", m_styleCls);
        if (vec.canConvert<UI_VECTYPE>())
            setText(vec.value<UI_VECTYPE>().at(i), m_editors[i]);
        else if (vec.canConvert<UI_VECSTRING>())
            setText(vec.value<UI_VECSTRING>().at(i), m_editors[i]);
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

UI_VECTYPE ZVecEditor::text() const 
{
    UI_VECTYPE vec;
    for (int i = 0; i < m_editors.size(); i++) {
        if (m_bFloat) 
            vec.append(m_editors[i]->text().toFloat());
        else
            vec.append(m_editors[i]->text().toInt());
    }
    return vec;
}

QVariant ZVecEditor::vec() const
{
	QVariant value;
    CURVES_DATA datas;
    UI_VECTYPE vec;
    UI_VECSTRING vecStr;
	for (int i = 0; i < m_editors.size(); i++)
	{
        if (m_bFloat) {
            if (m_editors[i]->property(g_keyFrame).canConvert<CURVE_DATA>()) 
            {
                CURVE_DATA data = m_editors[i]->property(g_keyFrame).value<CURVE_DATA>();
                QString key = UiHelper::getCurveKey(i);
                datas.insert(key, data);
            } 
            else 
            {
                bool bOK = false;
                float val = m_editors[i]->text().toFloat(&bOK);
                if (bOK && vecStr.isEmpty())
                    vec.append(val);
                else {
                    for (auto data : vec) {
                        vecStr.append(QString::number(data));
                    }
                    vec.clear();
                    vecStr.append(m_editors[i]->text());
                }
            }
        }
        else {
            bool bOK = false;
            float val = m_editors[i]->text().toInt(&bOK);
            if (bOK && vecStr.isEmpty())
                vec.append(val);
            else {
                for (auto data : vec) {
                    vecStr.append(QString::number(data));
                }
                vec.clear();
                vecStr.append(m_editors[i]->text());
            }
        }
	}
    if (vec.size() == m_editors.size()) 
    {
        value = QVariant::fromValue(vec);
    } 
    else if (vecStr.size() == m_editors.size()) 
    {
        value = QVariant::fromValue(vecStr);
    }
    else 
    {
        value = QVariant::fromValue(datas);
    }
    return value;
}

void ZVecEditor::setVec(const QVariant& vec, bool bFloat)
{
    int size = m_editors.size();
    if (vec.canConvert<UI_VECTYPE>())
        size = vec.value<UI_VECTYPE>().size();
    else if (vec.canConvert<UI_VECSTRING>())
        size = vec.value<UI_VECSTRING>().size();
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
            else if (vec.canConvert<UI_VECSTRING>())
                setText(vec.value<UI_VECSTRING>().at(i), m_editors[i]);
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
    QString text = UiHelper::variantToString(value);
    lineEdit->setText(text);
    lineEdit->setProperty(g_keyFrame, QVariant());
}

int ZVecEditor::getCurrentEditor() 
{
    QPoint pos = QCursor::pos();
    pos = mapFromGlobal(pos);
    for (int i = 0; i < m_editors.size(); i++)
    {
        if (m_editors.at(i)->geometry().contains(pos)) {
            return i;
        }
    }
    return -1;
}