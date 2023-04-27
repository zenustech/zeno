#include "zveceditoritem.h"
#include <zenomodel/include/uihelper.h>
#include <zenomodel/include/modeldata.h>
#include "../../style/zenostyle.h"


ZVecEditorItem::ZVecEditorItem(const UI_VECTYPE& vec, bool bFloat, LineEditParam param, QGraphicsScene* pScene, QGraphicsItem* parent, Qt::WindowFlags wFlags)
    : _base(parent)
    , m_bFloatVec(bFloat)
{
    initUI(vec, bFloat, pScene);
}

void ZVecEditorItem::initUI(const UI_VECTYPE& vec, bool bFloat, QGraphicsScene* pScene)
{
    for (int i = 0; i < m_editors.size(); i++)
    {
        delete m_editors[i];
    }
    m_editors.clear();

    ZGraphicsLayout* pLayout = new ZGraphicsLayout(true);
    pLayout->setContentsMargin(0, 0, 0, 0);
    pLayout->setSpacing(6);
    for (int i = 0; i < vec.size(); i++)
    {
        const QString& numText = QString::number(vec[i]);
        ZEditableTextItem* pLineEdit = new ZEditableTextItem(numText);

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

UI_VECTYPE ZVecEditorItem::vec() const
{
    UI_VECTYPE vec;
    for (auto editor : m_editors)
    {
        if (m_bFloatVec)
        {
            vec.append(editor->text().toFloat());
        }
        else
        {
            vec.append(editor->text().toInt());
        }
    }
    return vec;
}

void ZVecEditorItem::setVec(const UI_VECTYPE& vec, bool bFloat, QGraphicsScene* pScene)
{
    if (vec.size() != m_editors.size())
        return;

    for (int i = 0; i < vec.size(); i++)
    {
        m_editors[i]->setText(QString::number(vec[i]));
    }
}

void ZVecEditorItem::setVec(const UI_VECTYPE& vec)
{
    if (vec.size() != m_editors.size())
        return;

    for (int i = 0; i < vec.size(); i++)
    {
        m_editors[i]->setText(QString::number(vec[i]));
    }
}

bool ZVecEditorItem::isFloatType() const
{
    return m_bFloatVec;
}