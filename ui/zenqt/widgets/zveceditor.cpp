#include "zveceditor.h"
#include "style/zenostyle.h"
#include "util/uihelper.h"
#include "zlineedit.h"
#include "util/curveutil.h"
#include <zeno/utils/log.h>
#include "panel/zenoproppanel.h"
#include "zassert.h"
#include <zeno/core/IObject.h>
#include "zenoapplication.h"
#include "zenomainwindow.h"
#include "widgets/ztimeline.h"
#include "curvemap/zcurvemapeditor.h"


ZVecEditor::ZVecEditor(const zeno::vecvar& vec, bool bFloat, QString styleCls, QWidget* parent)
    : QWidget(parent)
    , m_bFloat(bFloat)
    , m_styleCls(styleCls)
    , m_hintlist(nullptr)
    , m_descLabel(nullptr)
    , m_vec(vec)
{
    m_deflSize = vec.size();
    initUI(m_vec);
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

void ZVecEditor::initUI(const zeno::vecvar& vecedit) {
    QHBoxLayout* pLayout = new QHBoxLayout;
    pLayout->setContentsMargins(0, 0, 0, 0);
    pLayout->setSpacing(5);

    int n = vecedit.size();
    m_editors.resize(n);
    for (int i = 0; i < n; i++)
    {
        m_editors[i] = new ZLineEdit;
        if (m_bFloat) {
            m_editors[i]->installEventFilter(this);
        }

        m_editors[i]->setNumSlider(UiHelper::getSlideStep("", m_bFloat ? zeno::types::gParamType_Float : zeno::types::gParamType_Int));
        //m_editors[i]->setFixedWidth(ZenoStyle::dpiScaled(64));
        m_editors[i]->setProperty("cssClass", m_styleCls);

        QString text = std::visit([](auto&& val) -> QString {
            using T = std::decay_t<decltype(val)>;
            if constexpr (std::is_same_v<T, int>) {
                return QString::number(val);
            }
            else if constexpr (std::is_same_v<T, float>) {
                return QString::number(val);
            }
            else if constexpr (std::is_same_v<T, std::string>) {
                return QString::fromStdString(val);
            }
            else {
                return "";
            }
        }, vecedit[i]);

        ZASSERT_EXIT(!text.isEmpty());
        m_editors[i]->setText(text);

        pLayout->addWidget(m_editors[i]);
        connect(m_editors[i], &ZLineEdit::editingFinished, this, [=]() {
            QString newText = m_editors[i]->text();
            if (!m_bFloat) {
                bool bConvert = false;
                int ival = newText.toInt(&bConvert);
                if (bConvert) {
                    m_vec[i] = ival;
                }
                else {
                    //可以尝试一下转float
                    float fval = newText.toFloat(&bConvert);
                    if (bConvert) {
                        ival = static_cast<int>(fval);
                        m_vec[i] = ival;
                    }
                    else {
                        //可能是别的表达式了，这时候直接套字符串进去就行
                        m_vec[i] = newText.toStdString();
                    }
                }
            }
            else {
                bool isCurves = std::holds_alternative<zeno::CurveData>(m_vec[i]);
                if (isCurves) { //k帧相关
                    std::string xKey = curve_util::getCurveKey(i).toStdString();
                    zeno::CurveData curvedata = std::get<zeno::CurveData>(m_vec[i]);
                    bool bConvert = false;
                    float fval = newText.toFloat(&bConvert);
                    if (!bConvert) {
                        fval = newText.toInt(&bConvert);
                        if (!bConvert) {    //string
                        }
                    } 
                    bool exist = false;
                    ZenoMainWindow* mainWin = zenoApp->getMainWindow();
                    ZASSERT_EXIT(mainWin);
                    ZTimeline* timeline = mainWin->timeline();
                    ZASSERT_EXIT(timeline);
                    for (int i = 0; i < curvedata.cpbases.size(); i++) {
                        if (curvedata.cpbases[i] == timeline->value()) {
                            curvedata.cpoints[i].v = fval;
                            exist = true;
                            break;
                        }
                    }
                    if (exist) {
                        m_vec[i] = curvedata;
                    } else {
                        m_editors[i]->setText(QString::number(curvedata.eval(timeline->value())));
                        return;
                    }
                }
                else {
                    bool bConvert = false;
                    float fval = newText.toFloat(&bConvert);
                    if (bConvert) {
                        m_vec[i] = fval;
                    }
                    else {
                        //可以尝试一下转int
                        int ival = newText.toInt(&bConvert);
                        if (bConvert) {
                            fval = ival;
                            m_vec[i] = fval;
                        }
                        else {
                            m_vec[i] = newText.toStdString();
                        }
                    }
                }
            }
            emit valueChanged(m_vec);
        });
    }
    setLayout(pLayout);
    setStyleSheet("ZVecEditor { background: transparent; } ");
}

bool ZVecEditor::isFloat() const
{
    return m_bFloat;
}

zeno::vecvar ZVecEditor::vec() const
{
    return m_vec;
}

void ZVecEditor::setVec(const zeno::vecvar& editVec, bool bFloat)
{
    int size = m_editors.size();

    if (bFloat != m_bFloat || editVec.size() != size)
    {
        //类型大小发生了变化，应该只有子图参数才能发生
        Q_ASSERT(m_nodeIdx.data(ROLE_NODETYPE) == zeno::Node_SubgraphNode);
        initUI(editVec);
    }
    else
    {
        for (int i = 0; i < size; i++)
        {
            QString text = std::visit([](auto&& val) -> QString {
                using T = std::decay_t<decltype(val)>;
                if constexpr (std::is_same_v<T, int>) {
                    return QString::number(val);
                }
                else if constexpr (std::is_same_v<T, float>) {
                    return QString::number(val);
                }
                else if constexpr (std::is_same_v<T, std::string>) {
                    return QString::fromStdString(val);
                }
                else {
                    return "";
                }
            }, editVec[i]);
            m_editors[i]->setText(text);
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

void ZVecEditor::setKeyFrame(const QStringList& keys)
{
    if (m_vec.empty()) {
        return;
    }
    bool isCurves = std::holds_alternative<zeno::CurveData>(m_vec[0]);
    zeno::CurvesData curvesdata;
    if (!isCurves) {
        for (int i = 0; i < m_editors.size(); i++) {
            auto key = curve_util::getCurveKey(i).toStdString();
            zeno::CurveData curve = zeno::CurveData();
            bool visible = keys.contains(QString::fromStdString(key));
            curve_util::getDelfCurveData(curve, m_editors[i]->text().toFloat(), visible, QString::fromStdString(key));
            curve.visible = visible;
            m_vec[i] = curve;
            curvesdata.keys.insert({ key, curve });
        }
    } else {
        for (int i = 0; i < m_editors.size(); i++) {
            auto key = curve_util::getCurveKey(i).toStdString();
            bool visible = keys.contains(QString::fromStdString(key));
            zeno::CurveData curve = std::get<zeno::CurveData>(m_vec[i]);
            if (visible) {
                if (!curve.visible) {
                    curve = zeno::CurveData();
                }
                curve_util::getDelfCurveData(curve, m_editors[i]->text().toFloat(), visible, QString::fromStdString(key));
                curve.visible = visible;
                m_vec[i] = curve;
                curvesdata.keys.insert({ key, curve });
            }
        }
    }
    curve_util::updateRange(curvesdata);
    emit valueChanged(m_vec);
}

void ZVecEditor::delKeyFrame(const QStringList& keys)
{
    ZenoMainWindow* mainWin = zenoApp->getMainWindow();
    ZASSERT_EXIT(mainWin);
    ZTimeline* timeline = mainWin->timeline();
    ZASSERT_EXIT(timeline);
    zeno::vecvar tmpvar = m_vec;
    int currEditorIdx = 0;
    for (int i = 0; i < m_editors.size(); i++) {
        if (keys.contains(curve_util::getCurveKey(i))) {
            currEditorIdx = i;
            break;
        }
    }
    zeno::CurveData curve = std::get<zeno::CurveData>(m_vec[currEditorIdx]);
    for (int i = 0; i < curve.cpbases.size(); i++) {
        int x = static_cast<int>(curve.cpbases[i]);
        if (x == timeline->value()) {
            bool otherEditorHasCurve = false;
            for (int i = 0; i < m_editors.size(); i++) {
                if (i != currEditorIdx) {
                    if (std::get<zeno::CurveData>(m_vec[i]).visible) {
                        otherEditorHasCurve = true;
                        break;
                    }
                }
            }
            if (curve.cpbases.size() > 1) {
                curve.cpbases.erase(curve.cpbases.begin() + i);
                curve.cpoints.erase(curve.cpoints.begin() + i);
                tmpvar[currEditorIdx] = curve;
                m_vec[currEditorIdx] = curve;
            }
            else {
                if (!otherEditorHasCurve) {
                    for (int i = 0; i < m_editors.size(); i++) {
                        zeno::CurveData c = std::get<zeno::CurveData>(m_vec[i]);
                        c.visible = false;
                        tmpvar[i] = c;
                        m_vec[i] = m_editors[i]->text().toFloat();
                    }
                } else {
                    zeno::CurveData c = std::get<zeno::CurveData>(m_vec[currEditorIdx]);
                    c.visible = false;
                    tmpvar[currEditorIdx] = c;
                    m_vec[currEditorIdx] = c;
                }
            }
            break;
        }
    }
    updateProperties(curve_util::getKeyFrameProperty(QVariant::fromValue(zeno::reflect::make_any<zeno::vecvar>(tmpvar))));
    emit valueChanged(m_vec);
}

void ZVecEditor::editKeyFrame(const QStringList& keys)
{
    if (keys.empty())
        return;
    std::string key = keys[0].toStdString();
    zeno::CurveData curve;
    int curveIdx = 0;
    for (int i = 0; i < m_vec.size(); i++) {
        if (key == curve_util::getCurveKey(i).toStdString()) {
            curveIdx = i;
            curve = std::get<zeno::CurveData>(m_vec[i]);
            break;
        }
    }
    ZCurveMapEditor* pEditor = new ZCurveMapEditor(true);
    connect(pEditor, &ZCurveMapEditor::finished, this, [&pEditor, &key, &keys, &curveIdx, this](int result) {
        zeno::CurvesData newCurves = pEditor->curves();
        if (newCurves.contains(key)) {
            m_vec[curveIdx] = newCurves[key];
            emit valueChanged(m_vec);
        }
        else {
            ZenoMainWindow* mainWin = zenoApp->getMainWindow();
            ZASSERT_EXIT(mainWin);
            ZTimeline* timeline = mainWin->timeline();
            ZASSERT_EXIT(timeline);
            zeno::CurveData curve = std::get<zeno::CurveData>(m_vec[curveIdx]);
            auto point = curve.cpoints[0];
            point.v = m_editors[curveIdx]->text().toFloat();
            curve.cpbases.clear();
            curve.cpoints.clear();
            curve.cpbases.push_back(timeline->value());
            curve.cpoints.push_back(point);
            curve.visible = false;
            m_vec[curveIdx] = curve;
            updateProperties(curve_util::getKeyFrameProperty(QVariant::fromValue(zeno::reflect::make_any<zeno::vecvar>(m_vec))));
            emit valueChanged(m_vec);
        }
    });
    zeno::CurvesData curves;
    curves.keys.insert({ key, curve });
    pEditor->setAttribute(Qt::WA_DeleteOnClose);
    pEditor->addCurves(curves);
    CURVES_MODEL models = pEditor->getModel();
    for (auto model : models) {
        for (int i = 0; i < model->rowCount(); i++) {
            model->setData(model->index(i, 0), true, ROLE_LOCKX);
        }
    }
    pEditor->exec();
}

void ZVecEditor::clearKeyFrame(const QStringList& keys)
{
    QVector<QString> properties;
    for (int i = 0; i < m_editors.size(); i++) {
        if (keys.contains(curve_util::getCurveKey(i))) {
            ZenoMainWindow* mainWin = zenoApp->getMainWindow();
            ZASSERT_EXIT(mainWin);
            ZTimeline* timeline = mainWin->timeline();
            ZASSERT_EXIT(timeline);
            zeno::CurveData curve = std::get<zeno::CurveData>(m_vec[i]);
            auto point = curve.cpoints[0];
            point.v = m_editors[i]->text().toFloat();
            curve.cpbases.clear();
            curve.cpoints.clear();
            curve.cpbases.push_back(timeline->value());
            curve.cpoints.push_back(point);
            curve.visible = false;
            m_vec[i] = curve;
        }
    }
    updateProperties(curve_util::getKeyFrameProperty(QVariant::fromValue(zeno::reflect::make_any<zeno::vecvar>(m_vec))));
    emit valueChanged(m_vec);
}

bool ZVecEditor::serKeyFrameStyle(QVariant qvar)
{
    QVariant newVal = qvar;
    if (!curve_util::getCurveValue(newVal))
        return false;
    zeno::vecvar vvar = newVal.value<zeno::vecvar>();

    for (int i = 0; i < m_editors.size(); i++) {
        QString text = UiHelper::editVariantToQString(vvar[i]);
        if (m_editors[i]->text() != text) {
            BlockSignalScope blkScop(m_editors[i]);
            m_editors[i]->setText(text);
        }
    }
    updateProperties(curve_util::getKeyFrameProperty(qvar));
    return true;
}
