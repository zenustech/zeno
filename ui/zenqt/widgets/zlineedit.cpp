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

            //º¯ÊýËµÃ÷
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
                        m_descLabel->setDesc(recommandInfo.func_args.func, 0);
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
            if (event->key() == Qt::Key_Down) {
                m_hintlist->setActive();
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
    QLineEdit::focusOutEvent(event);
}
