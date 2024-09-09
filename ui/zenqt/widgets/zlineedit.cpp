#include "zlineedit.h"
#include "znumslider.h"
#include "style/zenostyle.h"
#include <QSvgRenderer>
#include "curvemap/zcurvemapeditor.h"
#include "util/uihelper.h"
#include "panel/ZenoHintListWidget.h"
#include "panel/zenoproppanel.h"
#include "widgets/zlabel.h"
#include <zeno/formula/formula.h>
#include <util/log.h>
#include <zeno/utils/helper.h>
#include "util/apphelper.h"


ZLineEdit::ZLineEdit(QWidget* parent)
    : QLineEdit(parent)
    , m_pSlider(nullptr)
    , m_bShowingSlider(false)
    , m_bHasRightBtn(false)
    , m_pButton(nullptr)
    , m_bIconHover(false)
    , m_bShowHintList(true)
    , m_hintlist(nullptr)
    , m_descLabel(nullptr)

{
    init();
}

ZLineEdit::ZLineEdit(const QString& text, QWidget* parent)
    : QLineEdit(text, parent)
    , m_pSlider(nullptr)
    , m_bShowingSlider(false)
    , m_bHasRightBtn(false)
    , m_pButton(nullptr)
    , m_bIconHover(false)
    , m_hintlist(nullptr)
{
    init();
}

void ZLineEdit::sltHintSelected(QString itemSelected)
{
    hintSelectedSetText(itemSelected);
    setFocus();
    if (m_hintlist)
    {
        disconnect(m_hintlist, &ZenoHintListWidget::hintSelected, this, &ZLineEdit::sltHintSelected);
        disconnect(m_hintlist, &ZenoHintListWidget::escPressedHide, this, &ZLineEdit::sltSetFocus);
        disconnect(m_hintlist, &ZenoHintListWidget::resizeFinished, this, &ZLineEdit::sltSetFocus);
    }
}

void ZLineEdit::sltSetFocus()
{
    setFocus();
}

void ZLineEdit::init()
{
    connect(this, &ZLineEdit::editingFinished, this, [=]() {
        zeno::Formula fmla(text().toStdString(), "");
        int ret = fmla.parse();
        fmla.printSyntaxTree();
        emit textEditFinished();
    });
    connect(this, &QLineEdit::textChanged, this, [&](const QString& text) {
        if (m_hintlist && m_descLabel && hasFocus() && m_bShowHintList && m_nodeIdx.isValid())
        {
            QString txt = text.left(cursorPosition());
            QString nodePath = m_nodeIdx.data(ROLE_OBJPATH).toString();
            zeno::Formula fmla(txt.toStdString(), nodePath.toStdString());

            QFontMetrics metrics(this->font());
            const QPoint& parentGlobalPos = m_hintlist->getPropPanelPos();
            QPoint globalPos = this->mapToGlobal(QPoint(0, 0));
            globalPos.setX(globalPos.x() - parentGlobalPos.x() + metrics.width(txt));
            globalPos.setY(globalPos.y() - parentGlobalPos.y() + height());

            //函数说明
            int ret = fmla.parse();
            //fmla.printSyntaxTree();
            if (ret == 0 || fmla.getASTResult())
            {
                zeno::formula_tip_info recommandInfo = fmla.getRecommandTipInfo();
                if (recommandInfo.type == zeno::FMLA_TIP_FUNC_CANDIDATES ||
                    recommandInfo.type == zeno::FMLA_TIP_REFERENCE)
                {
                    QStringList items;
                    std::string candidateWord = recommandInfo.prefix;
                    for (auto& item : recommandInfo.func_candidats) {
                        items << QString::fromStdString(item);
                    }
                    for (auto& item : recommandInfo.ref_candidates) {
                        items << QString::fromStdString(item.nodename);
                    }
                    m_firstCandidateWord = QString::fromStdString(candidateWord);

                    if (items.size() == 0) {
                        if (m_hintlist->isVisible()) {
                            m_hintlist->hide();
                        }
                    }
                    else {
                        m_hintlist->setData(items);
                        m_hintlist->move(globalPos);
                        if (!m_hintlist->isVisible())
                        {
                            connect(m_hintlist, &ZenoHintListWidget::hintSelected, this, &ZLineEdit::sltHintSelected, Qt::UniqueConnection);
                            connect(m_hintlist, &ZenoHintListWidget::escPressedHide, this, &ZLineEdit::sltSetFocus, Qt::UniqueConnection);
                            connect(m_hintlist, &ZenoHintListWidget::resizeFinished, this, &ZLineEdit::sltSetFocus, Qt::UniqueConnection);
                            m_hintlist->show();
                            if (m_descLabel->isVisible()) {
                                m_descLabel->hide();
                            }
                        }
                        m_hintlist->resetCurrentItem();
                    }
                }
                else if (recommandInfo.type == zeno::FMLA_TIP_FUNC_ARGS)
                {
                    m_hintlist->hide();
                    if (recommandInfo.func_args.func.name.empty()) {
                        m_descLabel->hide();
                    }
                    else {
                        int pos = recommandInfo.func_args.argidx;
                        m_descLabel->setDesc(recommandInfo.func_args.func, recommandInfo.func_args.argidx - 1);
                        m_descLabel->move(globalPos);
                        if (!m_descLabel->isVisible()) {
                            m_descLabel->show();
                        }
                        m_descLabel->setCurrentFuncName(recommandInfo.func_args.func.name);
                    }
                }
                else if (recommandInfo.type == zeno::FMLA_NO_MATCH)
                {
                    m_hintlist->hide();
                    m_descLabel->hide();
                }
            }
            else if (m_descLabel->isVisible()) {
                m_descLabel->hide();
            }
        }
    });
}

void ZLineEdit::setShowingSlider(bool bShow)
{
    m_bShowingSlider = bShow;
}

bool ZLineEdit::showingSlider()
{
    return m_bShowingSlider;
}

void ZLineEdit::setIcons(const QString& icNormal, const QString& icHover)
{
    m_iconNormal = icNormal;
    m_iconHover = icHover;
    m_pButton = new QPushButton(this);
    m_pButton->setFixedSize(ZenoStyle::dpiScaled(20), ZenoStyle::dpiScaled(20));
    m_pButton->installEventFilter(this);
    QHBoxLayout *btnLayout = new QHBoxLayout(this);
    btnLayout->addStretch();
    btnLayout->addWidget(m_pButton);
    btnLayout->setAlignment(Qt::AlignRight);
    btnLayout->setContentsMargins(0, 0, 0, 0);
    connect(m_pButton, SIGNAL(clicked(bool)), this, SIGNAL(btnClicked()));
}

void ZLineEdit::hintSelectedSetText(QString text)
{
    BlockSignalScope scope(this);
    int newPos = cursorPosition() - m_firstCandidateWord.size() + text.size();
    QString txt = this->text();
    txt.replace(cursorPosition() - m_firstCandidateWord.size(), m_firstCandidateWord.size(), text);
    setText(txt);
    setCursorPosition(newPos);
}

void ZLineEdit::setHintListWidget(ZenoHintListWidget* hintlist, ZenoFuncDescriptionLabel* descLabl)
{
    m_hintlist = hintlist;
    m_descLabel = descLabl;
}

void ZLineEdit::setNodeIdx(const QModelIndex& index) {
    m_nodeIdx = index;
}

void ZLineEdit::setNumSlider(const QVector<qreal>& steps)
{
    if (steps.isEmpty())
        return;

    m_steps = steps;
    m_pSlider = new ZNumSlider(m_steps, this);
    m_pSlider->setWindowFlags(Qt::Window | Qt::FramelessWindowHint);
    m_pSlider->hide();

    connect(m_pSlider, &ZNumSlider::numSlided, this, [=](qreal val) {
        bool bOk = false;
        qreal num = this->text().toFloat(&bOk);
        if (bOk)
        {
            num = num + val;
            QString newText = QString::number(num);
            setText(newText);
            emit editingFinished();
        }
    });
    connect(m_pSlider, &ZNumSlider::slideFinished, this, [=]() {
        setShowingSlider(false);
        emit editingFinished();
    });
}

void ZLineEdit::mouseReleaseEvent(QMouseEvent* event)
{
    if (m_hintlist)
    {
        m_hintlist->setCurrentZlineEdit(this);
    }
    if (event->button() == Qt::MiddleButton && m_pSlider)
    {
        m_bShowHintList = true;

        m_pSlider->hide();
        setShowingSlider(false);
        event->accept();
        return;
    }
    QLineEdit::mouseReleaseEvent(event);
}

void ZLineEdit::mousePressEvent(QMouseEvent* event)
{
    if (event->button() == Qt::MiddleButton && m_pSlider) {
        m_bShowHintList = false;

        QPoint globalpos = mapToGlobal(event->pos());
        popupSlider();
        globalpos.setX(globalpos.x() - m_pSlider->width() / 2 - (hasFocus() ? width() : 0 ));
        globalpos.setY(globalpos.y() - m_pSlider->height() / 2);

        m_pSlider->move(globalpos);
        qApp->sendEvent(m_pSlider, event);
        event->accept();
        return;
    }
    QLineEdit::mousePressEvent(event);
}

void ZLineEdit::mouseMoveEvent(QMouseEvent* event)
{
    if (m_pSlider && m_pSlider->isVisible())
    {
        qApp->sendEvent(m_pSlider, event);
        return;
    }
    QLineEdit::mouseMoveEvent(event);
}

void ZLineEdit::popupSlider()
{
    if (!m_pSlider)
        return;

    QSize sz = m_pSlider->size();
    QRect rc = QApplication::desktop()->screenGeometry();
    static const int _yOffset = ZenoStyle::dpiScaled(20);

    QPoint pos = this->cursor().pos();
    pos.setY(std::min(pos.y(), rc.bottom() - sz.height() / 2 - _yOffset));
    pos -= QPoint(0, sz.height() / 2);

    setShowingSlider(true);

    m_pSlider->move(pos);
    m_pSlider->show();
    m_pSlider->activateWindow();
    m_pSlider->setFocus();
    m_pSlider->raise();
}

void ZLineEdit::keyPressEvent(QKeyEvent* event)
{
    if (m_hintlist && hasFocus() && m_bShowHintList)
    {
        if (m_hintlist->isVisible())
        {
            if (event->key() == Qt::Key_Down || event->key() == Qt::Key_Up) {
                bool bDown = event->key() == Qt::Key_Down;
                m_hintlist->onSwitchItemByKey(bDown);
                event->accept();
                return;
            }
            else if (event->key() == Qt::Key_Escape)
            {
                m_hintlist->hide();
                setFocus();
                disconnect(m_hintlist, &ZenoHintListWidget::hintSelected, this, &ZLineEdit::sltHintSelected);
                disconnect(m_hintlist, &ZenoHintListWidget::escPressedHide, this, &ZLineEdit::sltSetFocus);
                disconnect(m_hintlist, &ZenoHintListWidget::resizeFinished, this, &ZLineEdit::sltSetFocus);
                event->accept();
                return;
            }else if (event->key() == Qt::Key_Return || event->key() == Qt::Key_Enter)
            {
                if (m_hintlist->isVisible())
                {
                    m_hintlist->hide();
                    hintSelectedSetText(m_hintlist->getCurrentText());
                    event->accept();
                    setFocus();
                    return;
                }
            }
        }
        else if (m_descLabel && m_descLabel->isVisible())
        {
            if (event->key() == Qt::Key_Escape || event->key() == Qt::Key_Return || event->key() == Qt::Key_Enter) {
                m_descLabel->hide();
                setFocus();
                event->accept();
            }
        }
        else {
            if (event->key() == Qt::Key_Escape ||
                event->key() == Qt::Key_Return ||
                event->key() == Qt::Key_Enter) {
                this->clearFocus();
            }
        }
    }
    QLineEdit::keyPressEvent(event);
}

void ZLineEdit::paintEvent(QPaintEvent* event)
{
    QLineEdit::paintEvent(event);
    if (hasFocus())
    {
        QPainter p(this);
        QRect rc = rect();
        p.setPen(QColor("#4B9EF4"));
        p.setRenderHint(QPainter::Antialiasing, false);
        p.drawRect(rc.adjusted(0,0,-1,-1));
    }
}

void ZLineEdit::wheelEvent(QWheelEvent* event)
{
    if (hasFocus())
    {
        bool ok;
        double num = text().toDouble(&ok);
        if (ok)
        {
            if (event->delta() > 0)
            {
                num += 0.1;
            }
            else {
                num -= 0.1;
            }
            setText(QString::number(num));
        }
        event->accept();
        return;
    }
    QLineEdit::wheelEvent(event);
}

bool ZLineEdit::eventFilter(QObject *obj, QEvent *event) {
    if (obj == m_pButton) {
        if (event->type() == QEvent::Paint) {
            QSvgRenderer svgRender;
            QPainter painter(m_pButton);
            QRect rc = m_pButton->rect();
            if (m_bIconHover)
                svgRender.load(m_iconHover);
            else
                svgRender.load(m_iconNormal);
            svgRender.render(&painter, rc);
            return true;
        } else if (event->type() == QEvent::HoverEnter) {
            setCursor(QCursor(Qt::ArrowCursor));
            m_bIconHover = true;
        } else if (event->type() == QEvent::HoverLeave) {
            setCursor(QCursor(Qt::IBeamCursor));
            m_bIconHover = false;
        }
    }
    return QLineEdit::eventFilter(obj, event);
}

void ZLineEdit::focusOutEvent(QFocusEvent* event)
{
    if (m_hintlist && !m_hintlist->isVisible())
    {
        disconnect(m_hintlist, &ZenoHintListWidget::hintSelected, this, &ZLineEdit::sltHintSelected);
        disconnect(m_hintlist, &ZenoHintListWidget::escPressedHide, this, &ZLineEdit::sltSetFocus);
        disconnect(m_hintlist, &ZenoHintListWidget::resizeFinished, this, &ZLineEdit::sltSetFocus);
    }
    //右键显示keyframemenu导致的focusout不发出editfinish信号
    if (event->reason() == Qt::PopupFocusReason) {
        BlockSignalScope scp(this);
        QLineEdit::focusOutEvent(event);
    }
    else {
        QLineEdit::focusOutEvent(event);
    }
}



ZCoreParamLineEdit::ZCoreParamLineEdit(zeno::PrimVar var, zeno::ParamType targetType, QWidget* parent)
    : ZLineEdit(QString::fromStdString(zeno::editVariantToStr(var)), parent)
    , m_var(var)
    , m_targetType(targetType)
{
    connect(this, &ZLineEdit::textEditFinished, this, [=]() {
        QString newText = this->text();
        if (m_targetType == gParamType_Int) {
            bool bConvert = false;
            int ival = newText.toInt(&bConvert);
            if (bConvert) {
                m_var = ival;
            }
            else {
                //可以尝试一下转float
                float fval = newText.toFloat(&bConvert);
                if (bConvert) {
                    ival = static_cast<int>(fval);
                    m_var = ival;
                }
                else {
                    //可能是别的表达式了，这时候直接套字符串进去就行
                    m_var = newText.toStdString();
                }
            }
        }
        else if (m_targetType == gParamType_Float) {
            bool bConvert = false;
            float fval = newText.toFloat(&bConvert);
            if (bConvert) {
                m_var = fval;
            }
            else {
                //可以尝试一下转int
                int ival = newText.toInt(&bConvert);
                if (bConvert) {
                    fval = ival;
                    m_var = fval;
                }
                else {
                    m_var = newText.toStdString();
                }
            }
        }
        else if (m_targetType == gParamType_String) {
            m_var = newText.toStdString();
        }
        else if (m_targetType == gParamType_Curve) {    //k帧相关
            std::string xKey = "x";
            zeno::CurveData curvedata = std::get<zeno::CurveData>(m_var);

            float var = this->text().toFloat();
            bool exist = false;
            ZenoMainWindow* mainWin = zenoApp->getMainWindow();
            ZASSERT_EXIT(mainWin);
            ZTimeline* timeline = mainWin->timeline();
            ZASSERT_EXIT(timeline);
            for (int i = 0; i < curvedata.cpbases.size(); i++) {
                if (curvedata.cpbases[i] == timeline->value()) {
                    curvedata.cpoints[i].v = var;
                    exist = true;
                    break;
                }
            }
            if (exist) {
                m_var = curvedata;
            }
            else {
                setText(QString::number(curvedata.eval(timeline->value())));
                return;
            }
        }
        emit valueChanged(m_var);
    });
}

zeno::PrimVar ZCoreParamLineEdit::getPrimVariant() const
{
    return m_var;
}

void ZCoreParamLineEdit::setKeyFrame(const QStringList& keys)
{
    std::string xKey = "x";
    zeno::CurvesData curvesdata;
    float var = 0;
    if (m_targetType != gParamType_Curve) {
        switch (m_targetType)
        {
            case gParamType_Int:
            case gParamType_Float:
                var = this->text().toFloat();
                break;
            case gParamType_String:
                //获取formula结果
                break;
            default:
                break;
        }
        m_targetType = gParamType_Curve;
        curvesdata.keys.insert({xKey, zeno::CurveData()});
    }
    else {
        curvesdata.keys.insert({ xKey, std::get<zeno::CurveData>(m_var) });
        var = this->text().toFloat();
    }

    curve_util::getDelfCurveData(curvesdata.keys[xKey], var, true, QString::fromStdString(xKey));
    curve_util::updateRange(curvesdata);

    curvesdata.keys[xKey].visible = true;
    m_var = curvesdata.keys[xKey];

    emit valueChanged(m_var);
}

void ZCoreParamLineEdit::delKeyFrame(const QStringList& keys)
{
    zeno::CurveData curve = std::get<zeno::CurveData>(m_var);

    ZenoMainWindow* mainWin = zenoApp->getMainWindow();
    ZASSERT_EXIT(mainWin);
    ZTimeline* timeline = mainWin->timeline();
    ZASSERT_EXIT(timeline);
    for (int i = 0; i < curve.cpbases.size(); i++) {
        int x = static_cast<int>(curve.cpbases[i]);
        if (x == timeline->value()) {
            curve.cpbases.erase(curve.cpbases.begin() + i);
            curve.cpoints.erase(curve.cpoints.begin() + i);
            break;
        }
    }
    if (curve.cpbases.empty()) {
        bool bConvertInt = false, bConvertFloat = false;
        int ival = text().toInt(&bConvertInt);
        float fval = text().toFloat(&bConvertFloat);
        if (bConvertInt) {
            m_var = ival;
            m_targetType = gParamType_Int;
        } else if (bConvertFloat) {
            m_var = fval;
            m_targetType = gParamType_Float;
        } else {
            m_var = text().toStdString();
            m_targetType = gParamType_String;
        }
        setProperty(g_setKey, "null");
        this->style()->unpolish(this);
        this->style()->polish(this);
        update();
    }
    else {
        m_var = curve;
    }
    emit valueChanged(m_var);
}

void ZCoreParamLineEdit::editKeyFrame(const QStringList& keys)
{
    std::string xKey = "x";
    ZCurveMapEditor* pEditor = new ZCurveMapEditor(true);
    connect(pEditor, &ZCurveMapEditor::finished, this, [&pEditor, &xKey, &keys, this](int result) {
        zeno::CurvesData newCurves = pEditor->curves();
        if (newCurves.contains(xKey)) {
            m_var = newCurves[xKey];
            emit valueChanged(m_var);
        }
        else {
            clearKeyFrame(keys);
        }
    });
    zeno::CurveData curve = std::get<zeno::CurveData>(m_var);
    zeno::CurvesData curves;
    curves.keys.insert({xKey, curve});
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

void ZCoreParamLineEdit::clearKeyFrame(const QStringList& keys)
{
    bool bConvertInt = false, bConvertFloat = false;
    int ival = text().toInt(&bConvertInt);
    float fval = text().toFloat(&bConvertFloat);
    if (bConvertInt) {
        m_var = ival;
        m_targetType = gParamType_Int;
    }
    else if (bConvertFloat) {
        m_var = fval;
        m_targetType = gParamType_Float;
    }
    else {
        m_var = text().toStdString();
        m_targetType = gParamType_String;
    }
    setProperty(g_setKey, "null");
    this->style()->unpolish(this);
    this->style()->polish(this);
    update();
    emit valueChanged(m_var);
}

bool ZCoreParamLineEdit::serKeyFrameStyle(QVariant qvar)
{
    QVariant newVal = qvar;
    if (!curve_util::getCurveValue(newVal))
        return false;

    QString text = UiHelper::variantToString(newVal);
    if (this->text() != text) {
        setText(text);
    }
    QVector<QString> properties = curve_util::getKeyFrameProperty(qvar);
    setProperty(g_setKey, properties.first());
    this->style()->unpolish(this);
    this->style()->polish(this);
    update();
    return true;
}
