#include "zveceditor.h"
#include "style/zenostyle.h"
#include "util/uihelper.h"
#include "zlineedit.h"
#include "util/curveutil.h"
#include <zeno/utils/log.h>


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

    m_editors.resize(n);
    for (int i = 0; i < m_editors.size(); i++)
    {
        m_editors[i] = new ZLineEdit;
        if (m_bFloat) {
            m_editors[i]->installEventFilter(this);
        }

        m_editors[i]->setNumSlider(UiHelper::getSlideStep("", m_bFloat ? zeno::Param_Float : zeno::Param_Int));
        //m_editors[i]->setFixedWidth(ZenoStyle::dpiScaled(64));
        m_editors[i]->setProperty("cssClass", m_styleCls);
        if (vec.canConvert<UI_VECTYPE>())
            setText(vec.value<UI_VECTYPE>().at(i), m_editors[i]);
        else if (vec.canConvert<UI_VECSTRING>())
            setText(vec.value<UI_VECSTRING>().at(i), m_editors[i]);

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
    UI_VECTYPE vec;
    UI_VECSTRING vecStr;
	for (int i = 0; i < m_editors.size(); i++)
	{
        if (m_bFloat) {
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
                    QString str = m_editors[i]->text();
                    if (!bOK && !str.startsWith("="))
                        zeno::log_error("The formula '{}' need start with '='", str.toStdString());
                    vecStr.append(str);
                }
            }
        }
        else {
            bool bOK = false;
            int val = m_editors[i]->text().toInt(&bOK);
            if (bOK && vecStr.isEmpty())
                vec.append(val);
            else {
                for (auto data : vec) {
                    vecStr.append(QString::number(data));
                }
                vec.clear();
                QString str = m_editors[i]->text();
                if (!bOK && !str.startsWith("="))
                    zeno::log_error("The formula '{}' need start with '='", str.toStdString());
                vecStr.append(str);
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
    return value;
}

void ZVecEditor::setVec(const QVariant& vec, bool bFloat)
{
    int size = m_editors.size();
    if (vec.canConvert<UI_VECTYPE>())
        size = vec.value<UI_VECTYPE>().size();
    else if (vec.canConvert<UI_VECSTRING>())
        size = vec.value<UI_VECSTRING>().size();
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
        }
    }
}

void ZVecEditor::setText(const QVariant &value, ZLineEdit* lineEdit) 
{
    QString text = UiHelper::variantToString(value);
    lineEdit->setText(text);
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

void ZVecEditor::updateProperties(const QVector<QString>& properties)
{
    for (int i = 0; i < m_editors.size(); i++)
    {
        QString property;
        if (i >= properties.size())
        {
            property = properties.first();
        }
        else
        {
            property = properties.at(i);
        }
        m_editors[i]->setProperty(g_setKey, property);
        m_editors[i]->style()->unpolish(m_editors[i]);
        m_editors[i]->style()->polish(m_editors[i]);
        m_editors[i]->update();
    }
}