#include "zlineedit.h"
#include "znumslider.h"
#include "style/zenostyle.h"
#include <QSvgRenderer>
#include "curvemap/zcurvemapeditor.h"
#include "util/uihelper.h"
#include "zenoapplication.h"
#include "zenomainwindow.h"
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
    blockSignals(true);
    setText(itemSelected);
    blockSignals(false);
    setFocus();
    disconnect(&ZenoPropPanel::getHintListInstance(), &ZenoHintListWidget::hintSelected, this, &ZLineEdit::sltHintSelected);
    disconnect(&ZenoPropPanel::getHintListInstance(), &ZenoHintListWidget::escPressedHide, this, &ZLineEdit::sltSetFocus);
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
            ZenoHintListWidget* listView = &ZenoPropPanel::getHintListInstance();

            QFontMetrics metrics(this->font());
            const QPoint& mainWinGlobalPos = zenoApp->getMainWindow()->mapToGlobal(QPoint(0,0));
            QPoint globalPos = this->mapToGlobal(QPoint(0, 0));
            globalPos.setX(globalPos.x() - mainWinGlobalPos.x() + metrics.width(text));
            globalPos.setY(globalPos.y() - mainWinGlobalPos.y() + height());

            //≤‚ ‘
            QStringList items;
            items << "1111" << "2222" << "3333" << "4444" << "5555" << "6666" << "7777" << "8888" << "9999" << "1100";
            listView->setData(items);

            listView->move(globalPos);
            if (!listView->isVisible())
            {
                connect(&ZenoPropPanel::getHintListInstance(), &ZenoHintListWidget::hintSelected, this, &ZLineEdit::sltHintSelected, Qt::UniqueConnection);
                connect(&ZenoPropPanel::getHintListInstance(), &ZenoHintListWidget::escPressedHide, this, &ZLineEdit::sltSetFocus, Qt::UniqueConnection);
                listView->show();
                listView->clearCurrentItem();
            }
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
    QLineEdit::mouseReleaseEvent(event);
}

void ZLineEdit::keyPressEvent(QKeyEvent* event)
{
    if (hasFocus() && m_bShowHintList)
    {
        ZenoHintListWidget* listView = &ZenoPropPanel::getHintListInstance();
        if (listView->isVisible())
        {
            if (event->key() == Qt::Key_Down) {
                listView->setActive();
                event->accept();
                return;
            }
            else if (event->key() == Qt::Key_Escape)
            {
                listView->hide();
                setFocus();
                disconnect(&ZenoPropPanel::getHintListInstance(), &ZenoHintListWidget::hintSelected, this, &ZLineEdit::sltHintSelected);
                disconnect(&ZenoPropPanel::getHintListInstance(), &ZenoHintListWidget::escPressedHide, this, &ZLineEdit::sltSetFocus);
                event->accept();
                return;
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

void ZLineEdit::resizeEvent(QResizeEvent* event)
{
    if (ZenoPropPanel::getHintListInstance().isVisible())
    {
        ZenoPropPanel::getHintListInstance().hide();
    }
    QLineEdit::resizeEvent(event);
}
