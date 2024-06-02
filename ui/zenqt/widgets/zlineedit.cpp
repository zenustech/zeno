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
        zeno::Formula fmla(text().toStdString());
        int ret = fmla.parse();
        fmla.printSyntaxTree();
        emit textEditFinished();
    });
    connect(this, &QLineEdit::textChanged, this, [&](const QString& text) {
        if (m_hintlist && m_descLabel && hasFocus() && m_bShowHintList)
        {
            QString txt = text.left(cursorPosition());
            zeno::Formula fmla(txt.toStdString());

            QFontMetrics metrics(this->font());
            const QPoint& parentGlobalPos = m_hintlist->getPropPanelPos();
            QPoint globalPos = this->mapToGlobal(QPoint(0, 0));
            globalPos.setX(globalPos.x() - parentGlobalPos.x() + metrics.width(txt));
            globalPos.setY(globalPos.y() - parentGlobalPos.y() + height());

            //设置函数提示列表内容
            //不能直接采用正则表达式识别函数，还是要归到语法树，这样才能即得到函数名称，以及函数参数位置。
#if 0
            QStringList items;
            std::string candidateWord = "";
            for (auto& i : fmla.getHintList(txt.toStdString(), candidateWord)) {
                items << QString::fromStdString(i);
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
                return;
            }
#endif
            //函数说明
            int ret = fmla.parse();
            //fmla.printSyntaxTree();
            if (ret == 0 || fmla.getRoot())
            {
                std::optional<std::tuple<std::string, std::string, int>> optNameDescPos = fmla.getCurrFuncDescription();
                if (!optNameDescPos.has_value())
                {
                    if (m_descLabel->isVisible()) {
                        m_descLabel->hide();
                    }
                } else {
                    auto nameDescPos = optNameDescPos.value();
                    m_descLabel->setDesc(QString::fromStdString(std::get<1>(nameDescPos)), std::get<2>(nameDescPos));
                    //if (m_descLabel->getCurrentFuncName() != std::get<0>(nameDescPos)) {
                    //    m_descLabel->move(globalPos);
                    //}
                    m_descLabel->move(globalPos);
                    if (!m_descLabel->isVisible()) {
                        m_descLabel->show();
                    }
                    m_descLabel->setCurrentFuncName(std::get<0>(nameDescPos));
                }
            } else if (m_descLabel->isVisible()) {
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
        if (m_descLabel && m_descLabel->isVisible())
        {
            if (event->key() == Qt::Key_Escape || event->key() == Qt::Key_Return || event->key() == Qt::Key_Enter) {
                m_descLabel->hide();
                setFocus();
                event->accept();
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
