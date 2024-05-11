#include "zlineedit.h"
#include "znumslider.h"
#include "style/zenostyle.h"
#include <QSvgRenderer>
#include "curvemap/zcurvemapeditor.h"
#include "util/uihelper.h"
#include "panel/ZenoHintListWidget.h"
#include "panel/zenoproppanel.h"

ZLineEdit::ZLineEdit(QWidget* parent)
    : QLineEdit(parent)
    , m_bHasRightBtn(false)
    , m_pButton(nullptr)
    , m_bIconHover(false)
    , m_bShowHintList(true)

{
    init();
}

ZLineEdit::ZLineEdit(const QString& text, QWidget* parent)
    : QLineEdit(text, parent)
    , m_bHasRightBtn(false)
    , m_pButton(nullptr)
    , m_bIconHover(false)
{
    init();
}

void ZLineEdit::sltHintSelected(QString itemSelected)
{
    setText(itemSelected);
    setFocus();
    disconnect(&ZenoPropPanel::getHintListInstance(), &ZenoHintListWidget::hintSelected, this, &ZLineEdit::sltHintSelected);
    disconnect(&ZenoPropPanel::getHintListInstance(), &ZenoHintListWidget::escPressedHide, this, &ZLineEdit::sltSetFocus);
    disconnect(&ZenoPropPanel::getHintListInstance(), &ZenoHintListWidget::resizeFinished, this, &ZLineEdit::sltSetFocus);
}

void ZLineEdit::sltSetFocus()
{
    setFocus();
}

void ZLineEdit::init()
{
    connect(this, SIGNAL(editingFinished()), this, SIGNAL(textEditFinished()));
    connect(this, &QLineEdit::textChanged, this, [&](const QString& text) {
        if (hasFocus() && m_bShowHintList)
        {
            ZenoHintListWidget* hintlist = &ZenoPropPanel::getHintListInstance();
            QFontMetrics metrics(this->font());
            const QPoint& parentGlobalPos = hintlist->getPropPanelPos();
            QPoint globalPos = this->mapToGlobal(QPoint(0, 0));
            globalPos.setX(globalPos.x() - parentGlobalPos.x() + metrics.width(text));
            globalPos.setY(globalPos.y() - parentGlobalPos.y() + height());

            //≤‚ ‘
            QStringList items;
            items << "1111" << "2222" << "3333" << "4444" << "5555" << "6666" << "7777" << "8888" << "9999" << "1100";
            hintlist->setData(items);

            hintlist->move(globalPos);
            if (!hintlist->isVisible())
            {
                connect(hintlist, &ZenoHintListWidget::hintSelected, this, &ZLineEdit::sltHintSelected, Qt::UniqueConnection);
                connect(hintlist, &ZenoHintListWidget::escPressedHide, this, &ZLineEdit::sltSetFocus, Qt::UniqueConnection);
                connect(hintlist, &ZenoHintListWidget::resizeFinished, this, &ZLineEdit::sltSetFocus, Qt::UniqueConnection);
                hintlist->show();
            }
            hintlist->resetCurrentItem();
        }
    });
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

void ZLineEdit::mouseReleaseEvent(QMouseEvent* event)
{
    ZenoPropPanel::getHintListInstance().setCurrentZlineEdit(this);
    QLineEdit::mouseReleaseEvent(event);
}

void ZLineEdit::keyPressEvent(QKeyEvent* event)
{
    if (hasFocus() && m_bShowHintList)
    {
        ZenoHintListWidget* hintlist = &ZenoPropPanel::getHintListInstance();
        if (hintlist->isVisible())
        {
            if (event->key() == Qt::Key_Down) {
                hintlist->setActive();
                event->accept();
                return;
            }
            else if (event->key() == Qt::Key_Escape)
            {
                hintlist->hide();
                setFocus();
                disconnect(hintlist, &ZenoHintListWidget::hintSelected, this, &ZLineEdit::sltHintSelected);
                disconnect(hintlist, &ZenoHintListWidget::escPressedHide, this, &ZLineEdit::sltSetFocus);
                disconnect(hintlist, &ZenoHintListWidget::resizeFinished, this, &ZLineEdit::sltSetFocus);
                event->accept();
                return;
            }else if (event->key() == Qt::Key_Return || event->key() == Qt::Key_Enter)
            {
                ZenoHintListWidget* hintlist = &ZenoPropPanel::getHintListInstance();
                if (hintlist->isVisible())
                {
                    hintlist->hide();
                    setText(hintlist->getCurrentText());
                }
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
    if (!ZenoPropPanel::getHintListInstance().isVisible())
    {
        disconnect(&ZenoPropPanel::getHintListInstance(), &ZenoHintListWidget::hintSelected, this, &ZLineEdit::sltHintSelected);
        disconnect(&ZenoPropPanel::getHintListInstance(), &ZenoHintListWidget::escPressedHide, this, &ZLineEdit::sltSetFocus);
        disconnect(&ZenoPropPanel::getHintListInstance(), &ZenoHintListWidget::resizeFinished, this, &ZLineEdit::sltSetFocus);
    }
    QLineEdit::focusOutEvent(event);
}
