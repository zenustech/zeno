#include "zveceditor.h"
#include "style/zenostyle.h"
#include "util/uihelper.h"
#include "zlineedit.h"
#include "util/curveutil.h"
#include <zeno/utils/log.h>
#include "panel/zenoproppanel.h"
#include "zassert.h"
#include <zeno/core/IObject.h>


ZVecEditor::ZVecEditor(const zeno::reflect::Any& vec, zeno::ParamType paramType, int deflSize, QString styleCls, QWidget* parent)
    : QWidget(parent)
    , m_paramType(paramType)
    , m_bFloat(m_paramType == gParamType_Vec2f || m_paramType == gParamType_Vec3f || m_paramType == gParamType_Vec4f)
    , m_deflSize(deflSize)
    , m_styleCls(styleCls)
    , m_hintlist(nullptr)
    , m_descLabel(nullptr)
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
    else if (event->type() == QEvent::FocusIn)
    {
        for (int i = 0; i < m_editors.size(); i++) {
            if (m_editors[i] != watched) {
                m_editors[i]->hide();
            }
        }
    }
    else if (event->type() == QEvent::FocusOut)
    {
        if (m_hintlist && !m_hintlist->isVisible())
        {
            if (ZLineEdit* edit = qobject_cast<ZLineEdit*>(watched))
            {
                if (!edit->hasFocus() && !edit->showingSlider())
                {
                    for (int i = 0; i < m_editors.size(); i++) {
                        if (!m_editors[i]->isVisible())
                            m_editors[i]->show();
                    }
                }
            }
        }
    }
    else if (event->type() == QEvent::KeyPress)
    {
        if (QKeyEvent* e = static_cast<QKeyEvent*>(event))
        {
            if (e->key() == Qt::Key_Escape)
            {
                if (ZLineEdit* edit = qobject_cast<ZLineEdit*>(watched))
                {
                    if (m_hintlist && m_hintlist->isVisible()) {
                        m_hintlist->hide();
                    } else if (m_descLabel && m_descLabel->isVisible()) {
                        m_descLabel->hide();
                    }
                    else {
                        edit->clearFocus();
                        for (int i = 0; i < m_editors.size(); i++) {
                            if (m_editors[i] != watched) {
                                m_editors[i]->show();
                            }
                        }
                    }
                }
            }
            else if (e->key() == Qt::Key_Return || e->key() == Qt::Key_Enter)
            {
                if (ZLineEdit* edit = qobject_cast<ZLineEdit*>(watched))
                {
                    if (m_hintlist && m_hintlist->isVisible())
                    {
                        m_hintlist->hide();
                        edit->hintSelectedSetText(m_hintlist->getCurrentText());
                    } else if (m_descLabel && m_descLabel->isVisible()) {
                        m_descLabel->hide();
                    } else {
                        edit->clearFocus();
                        for (int i = 0; i < m_editors.size(); i++) {
                            if (m_editors[i] != watched) {
                                m_editors[i]->show();
                            }
                        }
                    }
                    return true;
                }
            }
        }
    }
    return QWidget::eventFilter(watched, event);
}

void ZVecEditor::initUI(const zeno::reflect::Any& anyVal) {
    ZASSERT_EXIT(anyVal.has_value());

    QHBoxLayout* pLayout = new QHBoxLayout;
    pLayout->setContentsMargins(0, 0, 0, 0);
    pLayout->setSpacing(5);

    QStringList vecLiteral;
    int n = m_deflSize;

    if (zeno::reflect::get_type<zeno::vec3f>() == anyVal.type()) {
        const auto& vec = zeno::reflect::any_cast<zeno::vec3f>(anyVal);
        vecLiteral = QStringList({ QString::number(vec[0]), QString::number(vec[1]), QString::number(vec[2]) });
    }
    else if (zeno::reflect::get_type<zeno::vec3i>() == anyVal.type()) {
        const auto& vec = zeno::reflect::any_cast<zeno::vec3i>(anyVal);
        vecLiteral = QStringList({ QString::number(vec[0]), QString::number(vec[1]), QString::number(vec[2]) });
    }
    else if (zeno::reflect::get_type<zeno::vec3s>() == anyVal.type()) {
        const auto& vec = zeno::reflect::any_cast<zeno::vec3s>(anyVal);
        vecLiteral = QStringList({ QString::fromStdString(vec[0]), QString::fromStdString(vec[1]), QString::fromStdString(vec[2]) });
    }
    else if (zeno::reflect::get_type<zeno::vec2f>() == anyVal.type()) {
        const auto& vec = zeno::reflect::any_cast<zeno::vec2f>(anyVal);
        vecLiteral = QStringList({ QString::number(vec[0]), QString::number(vec[1]) });
    }
    else if (zeno::reflect::get_type<zeno::vec2i>() == anyVal.type()) {
        const auto& vec = zeno::reflect::any_cast<zeno::vec2i>(anyVal);
        vecLiteral = QStringList({ QString::number(vec[0]), QString::number(vec[1]) });
    }
    else if (zeno::reflect::get_type<zeno::vec2s>() == anyVal.type()) {
        const auto& vec = zeno::reflect::any_cast<zeno::vec2s>(anyVal);
        vecLiteral = QStringList({ QString::fromStdString(vec[0]), QString::fromStdString(vec[1]) });
    }
    else if (zeno::reflect::get_type<zeno::vec4f>() == anyVal.type()) {
        const auto& vec = zeno::reflect::any_cast<zeno::vec4f>(anyVal);
        vecLiteral = QStringList({ QString::number(vec[0]), QString::number(vec[1]), QString::number(vec[2]), QString::number(vec[3]) });
    }
    else if (zeno::reflect::get_type<zeno::vec4i>() == anyVal.type()) {
        const auto& vec = zeno::reflect::any_cast<zeno::vec4i>(anyVal);
        vecLiteral = QStringList({ QString::number(vec[0]), QString::number(vec[1]), QString::number(vec[2]), QString::number(vec[3]) });
    }
    else if (zeno::reflect::get_type<zeno::vec4s>() == anyVal.type()) {
        const auto& vec = zeno::reflect::any_cast<zeno::vec4s>(anyVal);
        vecLiteral = QStringList({ QString::fromStdString(vec[0]), QString::fromStdString(vec[1]), QString::fromStdString(vec[2]), QString::fromStdString(vec[3]) });
    }

    ZASSERT_EXIT(vecLiteral.size() == n);
    m_editors.resize(n);
    for (int i = 0; i < m_editors.size(); i++)
    {
        m_editors[i] = new ZLineEdit;
        if (m_bFloat) {
            m_editors[i]->installEventFilter(this);
        }

        m_editors[i]->setNumSlider(UiHelper::getSlideStep("", m_bFloat ? zeno::types::gParamType_Float : zeno::types::gParamType_Int));
        //m_editors[i]->setFixedWidth(ZenoStyle::dpiScaled(64));
        m_editors[i]->setProperty("cssClass", m_styleCls);
        setText(vecLiteral.at(i), m_editors[i]);

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

zeno::reflect::Any ZVecEditor::vec() const
{
    QVariant value;
    QVector<float> vec;
    QVector<std::string> vecStr;
    int n = m_editors.size();
    for (int i = 0; i < n; i++)
    {
        if (m_bFloat)
        {
            bool bOK = false;
            float val = m_editors[i]->text().toFloat(&bOK);
            if (bOK && vecStr.isEmpty()) {
                vec.append(val);
            }
            else {
                for (auto data : vec) {
                    vecStr.append(QString::number(data).toStdString());
                }
                vec.clear();
                QString str = m_editors[i]->text();
                vecStr.append(str.toStdString());
            }
        }
        else
        {
            bool bOK = false;
            int val = m_editors[i]->text().toInt(&bOK);
            if (bOK && vecStr.isEmpty()) {
                vec.append(val);
            }
            else {
                for (auto data : vec) {
                    vecStr.append(QString::number(data).toStdString());
                }
                vec.clear();
                QString str = m_editors[i]->text();
                vecStr.append(str.toStdString());
            }
        }
    }
    if (vec.size() == n) 
    {
        switch (n)
        {
        case 2: return m_bFloat ? zeno::vec2f(vec[0], vec[1]) : zeno::vec2i(vec[0], vec[1]);
        case 3: return m_bFloat ? zeno::vec3f(vec[0], vec[1], vec[2]) : zeno::vec3i(vec[0], vec[1], vec[2]);
        case 4: return m_bFloat ? zeno::vec4f(vec[0], vec[1], vec[2], vec[3]) : zeno::vec4i(vec[0], vec[1], vec[2], vec[3]);
        }
    } 
    else if (vecStr.size() == n) 
    {
        value = QVariant::fromValue(vecStr);
    }
    ZASSERT_EXIT(false, zeno::reflect::Any::make_null());
    return zeno::reflect::Any::make_null();
}

void ZVecEditor::setVec(const zeno::reflect::Any& editVec)
{
    int size = m_editors.size();

    bool bFloat = false;
    zeno::ParamType newType = editVec.type().hash_code();
    switch (newType) {
    case gParamType_Vec2i: size = 2; bFloat = false; break;
    case gParamType_Vec2f: size = 2; bFloat = true; break;
    case gParamType_Vec3i: size = 3; bFloat = false; break;
    case gParamType_Vec3f: size = 3; bFloat = true; break;
    case gParamType_Vec4i: size = 4; bFloat = false; break;
    case gParamType_Vec4f: size = 4; bFloat = true; break;
    default:
        ZASSERT_EXIT(false);
    }

    if (bFloat != m_bFloat || size != m_editors.size())
    {
        //类型大小发生了变化，应该只有子图参数才能发生
        Q_ASSERT(m_nodeIdx.data(ROLE_NODETYPE) == zeno::Node_SubgraphNode);
        initUI(editVec);
    }
    else
    {
        for (int i = 0; i < m_editors.size(); i++) 
        {
            switch (newType) {
            case gParamType_Vec2i: {
                int val = zeno::reflect::any_cast<zeno::vec2i>(editVec)[i];
                m_editors[i]->setText(QString::number(val));
                break;
            }
            case gParamType_Vec2f: {
                float val = zeno::reflect::any_cast<zeno::vec2f>(editVec)[i];
                m_editors[i]->setText(QString::number(val));
                break;
            }
            case gParamType_Vec2s: {
                std::string val = zeno::reflect::any_cast<zeno::vec2s>(editVec)[i];
                m_editors[i]->setText(QString::fromStdString(val));
                break;
            }
            case gParamType_Vec3i: {
                int val = zeno::reflect::any_cast<zeno::vec3i>(editVec)[i];
                m_editors[i]->setText(QString::number(val));
                break;
            }
            case gParamType_Vec3f: {
                float val = zeno::reflect::any_cast<zeno::vec3f>(editVec)[i];
                m_editors[i]->setText(QString::number(val));
                break;
            }
            case gParamType_Vec3s: {
                std::string val = zeno::reflect::any_cast<zeno::vec2s>(editVec)[i];
                m_editors[i]->setText(QString::fromStdString(val));
                break;
            }
            case gParamType_Vec4i: {
                int val = zeno::reflect::any_cast<zeno::vec4i>(editVec)[i];
                m_editors[i]->setText(QString::number(val));
                break;
            }
            case gParamType_Vec4f: {
                float val = zeno::reflect::any_cast<zeno::vec4f>(editVec)[i];
                m_editors[i]->setText(QString::number(val));
                break;
            }
            case gParamType_Vec4s: {
                std::string val = zeno::reflect::any_cast<zeno::vec4s>(editVec)[i];
                m_editors[i]->setText(QString::fromStdString(val));
                break;
            }
            }
        }
    }
}

void ZVecEditor::showNoFocusLineEdits(QWidget* lineEdit)
{
    if (lineEdit)
    {
        for (int i = 0; i < m_editors.size(); i++) {
            if (m_editors[i] == lineEdit)
                return;
        }
        for (int i = 0; i < m_editors.size(); i++) {
            if (!m_editors[i]->isVisible())
                m_editors[i]->show();
        }
    }
}

void ZVecEditor::setText(const QString& text, ZLineEdit* lineEdit)
{
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

void ZVecEditor::setNodeIdx(const QModelIndex& index)
{
    m_nodeIdx = index;
    for (int i = 0; i < m_editors.size(); i++)
    {
        m_editors[i]->setNodeIdx(m_nodeIdx);
    }
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

void ZVecEditor::setHintListWidget(ZenoHintListWidget* hintlist, ZenoFuncDescriptionLabel* descLabl)
{
    m_hintlist = hintlist;
    m_descLabel = descLabl;
    for (int i = 0; i < m_editors.size(); i++) {
        m_editors[i]->setHintListWidget(hintlist, descLabl);
    }
    connect(m_hintlist, &ZenoHintListWidget::clickOutSideHide, this, &ZVecEditor::showNoFocusLineEdits);
}
