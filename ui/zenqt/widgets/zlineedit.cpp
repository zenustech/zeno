#include "zlineedit.h"
#include "znumslider.h"
#include "style/zenostyle.h"
#include <QSvgRenderer>
#include "curvemap/zcurvemapeditor.h"
#include "util/uihelper.h"

ZLineEdit::ZLineEdit(QWidget* parent)
    : QLineEdit(parent)
    , m_bHasRightBtn(false)
    , m_pButton(nullptr)
    , m_bIconHover(false)

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

void ZLineEdit::init()
{
    connect(this, SIGNAL(editingFinished()), this, SIGNAL(textEditFinished()));
    //connect(this, &ZLineEdit::textChanged, this, &ZLineEdit::editingFinished);
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
    QLineEdit::mouseReleaseEvent(event);
}

void ZLineEdit::keyPressEvent(QKeyEvent* event)
{
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
