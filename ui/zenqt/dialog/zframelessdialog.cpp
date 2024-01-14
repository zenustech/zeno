#include "zframelessdialog.h"
#include "style/zenostyle.h"


ZFramelessDialog::ZFramelessDialog(QWidget* parent) : QDialog(parent)
{
    setAttribute(Qt::WA_QuitOnClose);
    setWindowFlags(this->windowFlags() | Qt::FramelessWindowHint);
    initTitleWidget();
    setStyleSheet("ZFramelessDialog{background-color: #22262B;}");
}

ZFramelessDialog::~ZFramelessDialog()
{
}

void ZFramelessDialog::setMainWidget(QWidget* pWidget)
{
    if (this->layout())
    {
        this->layout()->addWidget(pWidget);
    }
}

void ZFramelessDialog::setTitleIcon(const QIcon& icon)
{
    QPixmap pixmap = icon.pixmap(ZenoStyle::dpiScaledSize(QSize(20, 20)));
    m_pLbIcon->setPixmap(pixmap);
}

void ZFramelessDialog::setTitleText(const QString& text)
{
    m_pLbTitle->setText(text);
}

void ZFramelessDialog::initTitleWidget()
{
    m_pLbIcon = new QLabel(this);
    m_pLbIcon->setFixedSize(ZenoStyle::dpiScaledSize(QSize(20, 20)));

    m_pLbTitle = new QLabel(this);

    QPushButton* pBtnClose = new QPushButton(this);
    pBtnClose->setObjectName("closebtn");
    pBtnClose->setFixedSize(ZenoStyle::dpiScaledSize(QSize(20, 20)));
    connect(pBtnClose, &QPushButton::clicked, this, &ZFramelessDialog::close);

    QHBoxLayout* pTitleLayout = new QHBoxLayout;
    pTitleLayout->addWidget(m_pLbIcon, 0, Qt::AlignCenter);
    pTitleLayout->addSpacing(ZenoStyle::dpiScaled(4));
    pTitleLayout->addWidget(m_pLbTitle, 0, Qt::AlignCenter);
    pTitleLayout->addStretch();
    pTitleLayout->addWidget(pBtnClose, 0, Qt::AlignCenter);
    qreal margin = ZenoStyle::dpiScaled(8);
    pTitleLayout->setContentsMargins(margin, margin, margin, margin);

    QWidget* pTitleWidget = new QWidget(this);
    pTitleWidget->setLayout(pTitleLayout);
    pTitleWidget->setAutoFillBackground(true);
    QPalette pal = palette();
    pal.setColor(QPalette::Window, QColor("#121417"));
    pTitleWidget->setPalette(pal);
    pTitleWidget->setFixedHeight(ZenoStyle::dpiScaled(36));

    QVBoxLayout* pLayout = new QVBoxLayout(this);
    pLayout->addWidget(pTitleWidget);
    pLayout->setMargin(0);
}

void ZFramelessDialog::mousePressEvent(QMouseEvent* event)
{
    m_movePos = event->pos();
}

void ZFramelessDialog::mouseReleaseEvent(QMouseEvent* event)
{
    m_movePos = QPoint();
}

void ZFramelessDialog::mouseMoveEvent(QMouseEvent* event)
{
    if (!m_movePos.isNull())
    {
        move(event->globalPos() - m_movePos);
    }
}

void ZFramelessDialog::keyPressEvent(QKeyEvent* e)
{
    if (e->key() == Qt::Key_Enter || e->key() == Qt::Key_Return)
    {
        return;
    }
    QDialog::keyPressEvent(e);
}