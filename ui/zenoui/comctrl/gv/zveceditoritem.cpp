#include "zveceditoritem.h"
#include <zenomodel/include/uihelper.h>
#include <zenomodel/include/modeldata.h>
#include "../../style/zenostyle.h"


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
    else
        n = vec.value<CURVES_DATA>().size();
    for (int i = 0; i < n; i++)
    {
        ZEditableTextItem *pLineEdit = nullptr; 
        if (bFloat)
            pLineEdit = new ZFloatEditableTextItem;
        else
            pLineEdit = new ZEditableTextItem;
        if (vec.canConvert<UI_VECTYPE>()) {
            UI_VECTYPE tmp = vec.value<UI_VECTYPE>();
            pLineEdit->setText(QString::number(vec.value<UI_VECTYPE>().at(i)));
        } else if (vec.canConvert<CURVES_DATA>()) {
            CURVES_DATA curves = vec.value<CURVES_DATA>();
            QString key = UiHelper::getCurveKey(i);
            if (curves.contains(key)) {
                pLineEdit->setProperty(g_keyFrame, QVariant::fromValue(curves[key]));
            }
        }


        pLineEdit->setData(GVKEY_SIZEHINT, ZenoStyle::dpiScaledSize(QSizeF(64, 24)));
        pLineEdit->setData(GVKEY_SIZEPOLICY, QSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed));

        if (bFloat)
            pLineEdit->setValidator(new QDoubleValidator);
        else
            pLineEdit->setValidator(new QIntValidator);

        pLineEdit->setNumSlider(pScene, UiHelper::getSlideStep("", bFloat ? CONTROL_FLOAT : CONTROL_INT));
        m_editors.append(pLineEdit);
        connect(pLineEdit, SIGNAL(editingFinished()), this, SIGNAL(editingFinished()));

        pLayout->addItem(pLineEdit);
    }
    setLayout(pLayout);
}

QVariant ZVecEditorItem::vec() const
{
    QVariant value;
    CURVES_DATA datas;
    UI_VECTYPE vec;
    bool bKeyFrame = false;
    for (int i = 0; i < m_editors.size(); i++)
    {
        if (m_bFloatVec)
        {
            if (m_editors[i]->property(g_keyFrame).canConvert<CURVE_DATA>()) {
                bKeyFrame = true;
                CURVE_DATA data = m_editors[i]->property(g_keyFrame).value<CURVE_DATA>();
                datas.insert(data.key, data);
            } else {
                vec.append(m_editors[i]->text().toFloat());
            }
        }
        else
        {
            vec.append(m_editors[i]->text().toInt());
        }
    }
    if (!bKeyFrame) {
        value = QVariant::fromValue(vec);
    } else {
        value = QVariant::fromValue(datas);
    }
    return value;
}

void ZVecEditorItem::setVec(const QVariant& vec, bool bFloat, QGraphicsScene* pScene)
{
    /*if (vec.size() != m_editors.size())
        return;*/

    for (int i = 0; i < m_editors.size(); i++)
    {
        if (vec.canConvert<UI_VECTYPE>())
            m_editors[i]->setText(QString::number(vec.value<UI_VECTYPE>().at(i)));
        else if (vec.canConvert<CURVES_DATA>()) {
            CURVES_DATA curves = vec.value<CURVES_DATA>();
            QString key = UiHelper::getCurveKey(i);
            if (curves.contains(key)) {
                m_editors[i]->setProperty(g_keyFrame, QVariant::fromValue(curves[key]));
            }
        }
    }
}

void ZVecEditorItem::setVec(const QVariant& vec)
{
    //if (vec.size() != m_editors.size())
    //    return;

    for (int i = 0; i < m_editors.size(); i++)
    {
        if (vec.canConvert<UI_VECTYPE>())
            m_editors[i]->setText(QString::number(vec.value<UI_VECTYPE>().at(i)));
        else if (vec.canConvert<CURVES_DATA>()) {
            CURVES_DATA curves = vec.value<CURVES_DATA>();
            QString key = UiHelper::getCurveKey(i);
            if (curves.contains(key)) {
                m_editors[i]->setProperty(g_keyFrame, QVariant::fromValue(curves[key]));
            }
        }
    }
}

bool ZVecEditorItem::isFloatType() const
{
    return m_bFloatVec;
}