#include "zveceditoritem.h"
#include <zenomodel/include/uihelper.h>
#include <zenomodel/include/modeldata.h>
#include "../../style/zenostyle.h"
#include <zenomodel/include/curveutil.h>


ZVecEditorItem::ZVecEditorItem(const QVariant& vec, bool bFloat, LineEditParam param, QGraphicsScene* pScene, QGraphicsItem* parent, Qt::WindowFlags wFlags)
    : _base(parent)
    , m_bFloatVec(bFloat)
{
    initUI(vec, bFloat, pScene);
}

void ZVecEditorItem::initUI(const QVariant& vec, bool bFloat, QGraphicsScene* pScene)
{
    for (int i = 0; i < m_editors.size(); i++)
    {
        delete m_editors[i];
    }
    m_editors.clear();

    ZGraphicsLayout* pLayout = new ZGraphicsLayout(true);
    pLayout->setContentsMargin(0, 0, 0, 0);
    pLayout->setSpacing(6);
    int n = 0;
    if (vec.canConvert<UI_VECTYPE>())
        n = vec.value<UI_VECTYPE>().size();
    else if (vec.canConvert<UI_VECSTRING>())
        n = vec.value<UI_VECSTRING>().size();
    for (int i = 0; i < n; i++)
    {
        ZEditableTextItem *pLineEdit = new ZEditableTextItem;
        if (vec.canConvert<UI_VECTYPE>() || vec.canConvert<UI_VECSTRING>()) {
            QString text;
            if (vec.canConvert<UI_VECTYPE>())
                text = UiHelper::variantToString(vec.value<UI_VECTYPE>().at(i));
            else
                text = vec.value<UI_VECSTRING>().at(i);
            pLineEdit->setText(text);
        }

        pLineEdit->setData(GVKEY_SIZEHINT, ZenoStyle::dpiScaledSize(QSizeF(64, 24)));
        pLineEdit->setData(GVKEY_SIZEPOLICY, QSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed));

        if (bFloat && n != 3)
            pLineEdit->setValidator(new QDoubleValidator);
        else if (!bFloat)
            pLineEdit->setValidator(new QIntValidator);

        pLineEdit->setNumSlider(pScene, UiHelper::getSlideStep("", bFloat ? CONTROL_FLOAT : CONTROL_INT));
        m_editors.append(pLineEdit);
        connect(pLineEdit, SIGNAL(editingFinished()), this, SIGNAL(editingFinished()));

        pLayout->addItem(pLineEdit);
    }
    setLayout(pLayout);
}

UI_VECSTRING ZVecEditorItem::vecStr() const
{
    UI_VECSTRING vecStr;
    for (int i = 0; i < m_editors.size(); i++)
    {
        vecStr.append(m_editors[i]->text());
    }
    return vecStr;
}

QVariant ZVecEditorItem::vec() const
{
    QVariant value;
    UI_VECTYPE vec;
    UI_VECSTRING vecStr;
    for (int i = 0; i < m_editors.size(); i++)
    {
        if (m_bFloatVec)
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
        else
        {
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
    if (vec.size() == m_editors.size()) {
        value = QVariant::fromValue(vec);
    } else if (vecStr.size() == m_editors.size()) {
        value = QVariant::fromValue(vecStr);
    }
    return value;
}

void ZVecEditorItem::setVec(const QVariant& vec, bool bFloat, QGraphicsScene* pScene)
{
    /*if (vec.size() != m_editors.size())
        return;*/

    for (int i = 0; i < m_editors.size(); i++)
    {
        if (vec.canConvert<UI_VECTYPE>() || vec.canConvert<UI_VECSTRING>()) {
            QString text;
            if (vec.canConvert<UI_VECTYPE>())
                text = UiHelper::variantToString(vec.value<UI_VECTYPE>().at(i));
            else
                text = vec.value<UI_VECSTRING>().at(i);
            m_editors[i]->setText(text);
        }
    }
}

void ZVecEditorItem::setVec(const QVariant& vec)
{
    //if (vec.size() != m_editors.size())
    //    return;

    for (int i = 0; i < m_editors.size(); i++)
    {
        if (vec.canConvert<UI_VECTYPE>() || vec.canConvert<UI_VECSTRING>()) {
            QString text;
            if (vec.canConvert<UI_VECTYPE>())
                text = UiHelper::variantToString(vec.value<UI_VECTYPE>().at(i));
            else
                text = vec.value<UI_VECSTRING>().at(i);
            m_editors[i]->setText(text);
        }
    }
}

bool ZVecEditorItem::isFloatType() const
{
    return m_bFloatVec;
}

QString ZVecEditorItem::findElemByControl(ZEditableTextItem* pElem) const
{
    int idx = m_editors.indexOf(pElem);
    if (idx == 0) {
        return "x";
    }
    else if (idx == 1) {
        return "y";
    }
    else if (idx == 2) {
        return "z";
    }
    else if (idx == 3) {
        return "w";
    }
    else {
        return "";
    }
}

void ZVecEditorItem::updateProperties(const QVector<QString>& properties)
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
    }
}