#include "ZenoHintListWidget.h"
#include "style/zenostyle.h"
#include "panel/zenoproppanel.h"
#include "zenoapplication.h"
#include "zenomainwindow.h"
#include "zassert.h"

ZenoHintListWidget::ZenoHintListWidget(ZenoPropPanel* panel)
    : QWidget(panel)
    , m_listView(new QListView(this))
    , m_model(new QStringListModel(this))
    , m_parentPropanel(panel)
{
    //setMinimumSize({ minWidth,minHeight });
    setWindowFlags(windowFlags() | Qt::FramelessWindowHint| Qt::WindowStaysOnTopHint);

    QVBoxLayout* pLayout = new QVBoxLayout(this);
    if (pLayout)
    {
        pLayout->addWidget(m_listView);
        pLayout->setContentsMargins(SideLength/2, SideLength/2, SideLength/2, SideLength/2);
    }
    setLayout(pLayout);

    m_button = new TriangleButton("", this);
    m_button->installEventFilter(this);
    m_button->move(width() - SideLength, height() - SideLength);

    m_listView->installEventFilter(this);
    m_listView->setModel(m_model);
    m_listView->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_listView->setStyleSheet("border: 0px; font: 14pt Microsoft Sans Serif");
    connect(m_listView, SIGNAL(doubleClicked(const QModelIndex&)), this, SLOT(sltItemSelect(const QModelIndex&)));

    qApp->installEventFilter(this);
    this->hide();
};

void ZenoHintListWidget::setData(QStringList items) {
    m_model->setStringList(items);

    int maxSize = 0, maxItemIdx = -1;
    for (int i = 0; i < items.size(); ++i) {
        if (items.at(i).size() > maxSize) {
            maxSize = items.at(i).size();
            maxItemIdx = i;
        }
    }
    if (maxItemIdx != -1) {
        QFontMetrics lineditFontMetric(m_listView->font());
        setGeometry(x(), y(), ZenoStyle::dpiScaled(lineditFontMetric.horizontalAdvance(items.at(maxItemIdx)) + 50), height());
        m_button->move(width() - SideLength, height() - SideLength);
    }
};

void ZenoHintListWidget::onSwitchItemByKey(bool bDown) {
    //m_listView->setFocus(); 没必要获得焦点，让它留在编辑器上。
    QItemSelectionModel* selModel = m_listView->selectionModel();
    int r = selModel->currentIndex().row();
    if (bDown && r < m_model->rowCount() - 1) {
        r++;
    }
    else if (!bDown && r > 0) {
        r--;
    }
    const QModelIndex& idx = m_model->index(r, 0);
    if (idx.isValid())
        selModel->setCurrentIndex(idx, QItemSelectionModel::SelectCurrent);
};

void ZenoHintListWidget::resetCurrentItem()
{
    QItemSelectionModel* selModel = m_listView->selectionModel();
    const QModelIndex& first = m_model->index(0, 0);
    if (first.isValid())
        selModel->setCurrentIndex(first, QItemSelectionModel::SelectCurrent);
}

void ZenoHintListWidget::clearCurrentItem() {
    QItemSelectionModel* selModel = m_listView->selectionModel();
    selModel->clearCurrentIndex();
    selModel->clearSelection();
};

void ZenoHintListWidget::resetSize()
{
    setGeometry(x(), y(), minWidth, minHeight);
    m_button->move(width() - SideLength, height() - SideLength);
}

QString ZenoHintListWidget::getCurrentText()
{
    QItemSelectionModel* selModel = m_listView->selectionModel();
    const QModelIndex& curr = selModel->currentIndex();
    if (curr.isValid())
        return curr.data(Qt::DisplayRole).toString();
    return "";
}

QPoint ZenoHintListWidget::calculateNewPos(QWidget* widgetToFollow, const QString& txt)
{
    QFontMetrics metrics(widgetToFollow->font());
    if (QWidget* m_hintlistParent = qobject_cast<QWidget*>(this->parent())) {
        QPoint newpos = widgetToFollow->mapTo(m_hintlistParent, QPoint(0, 0));
        int parentwidth = m_hintlistParent->width();
        int txtwidth = metrics.width(txt);
        if (parentwidth < newpos.x() + txtwidth + this->width()) {
            newpos.setX(parentwidth - this->width());
        }
        else {
            newpos.setX(newpos.x() + txtwidth);
        }
        int parentheight = m_hintlistParent->height();
        if (parentheight < newpos.y() + widgetToFollow->height() + this->height()) {
            newpos.setY(newpos.y() - this->height());
        }
        else {
            newpos.setY(newpos.y() + widgetToFollow->height());
        }
        return newpos;
    }
    return {0,0};
}

void ZenoHintListWidget::updateParent()
{
    auto mainwin = zenoApp->getMainWindow();
    ZASSERT_EXIT(mainwin);
    if (mainwin->propPanelIsFloating(m_parentPropanel)) {
        setParent(m_parentPropanel);
    }
    else {
        setParent(mainwin);
    }
}

void ZenoHintListWidget::sltItemSelect(const QModelIndex& selectedIdx) {
    if (selectedIdx.isValid())
    {
        this->hide();
        emit hintSelected(selectedIdx.data(Qt::DisplayRole).toString());
    }
    else {
        QItemSelectionModel* selModel = m_listView->selectionModel();
        auto currentIdex = selModel->currentIndex();
        if (currentIdex.isValid()) {
            this->hide();
            emit hintSelected(selModel->currentIndex().data(Qt::DisplayRole).toString());
        }
    }
};

bool ZenoHintListWidget::eventFilter(QObject* watched, QEvent* event)
{
    if (watched == m_button)
    {
        if (event->type() == QEvent::Enter)
        {
            if (!m_resizing)
            {
                m_resizing = !m_resizing;
                setCursor(Qt::SizeFDiagCursor);
                return true;
            }
        }
        else if (event->type() == QEvent::Leave)
        {
            if (m_resizing)
            {
                m_resizing = !m_resizing;
                setCursor(Qt::ArrowCursor);
                return true;
            }
        }
    }
    if (watched == m_listView)
    {
        if (event->type() == QEvent::KeyPress)
        {
            if (QKeyEvent* e = static_cast<QKeyEvent*>(event))
            {
                if (e->key() == Qt::Key_Enter || e->key() == Qt::Key_Return)
                {
                    sltItemSelect(QModelIndex());
                    return true;
                }
                else if (e->key() == Qt::Key_Escape)
                {
                    this->hide();
                    emit escPressedHide();
                    return true;
                }
            }
        }
    }
    if (this->isVisible())
    {
        if (event->type() == QEvent::MouseButtonPress)
        {
            if (QMouseEvent* e = static_cast<QMouseEvent*>(event))
            {
                if (QWidget* wid = qobject_cast<QWidget*>(watched)) //点击区域不在内部则hide
                {
                    const QPoint& globalPos = wid->mapToGlobal(e->pos());
                    const QPoint& lefttop = mapToGlobal(QPoint(0, 0));
                    const QPoint& rightbottom = mapToGlobal(QPoint(width(), height()));
                    if (globalPos.x() < lefttop.x() || globalPos.x() > rightbottom.x() || globalPos.y() < lefttop.y() || globalPos.y() > rightbottom.y())
                    {
                        hide();
                        emit clickOutSideHide(wid);
                    }
                }
            }
        }
    }
    return QWidget::eventFilter(watched, event);
}

void ZenoHintListWidget::mouseMoveEvent(QMouseEvent* event)
{
    if (m_resizing)
    {
        setGeometry(x(), y(), event->pos().x(), event->pos().y());
        m_button->move(m_listView->width(), m_listView->height());
    }
}

void ZenoHintListWidget::mouseReleaseEvent(QMouseEvent* event)
{
    if (m_resizing)
    {
        emit resizeFinished();
    }
    QWidget::mouseReleaseEvent(event);
}

void ZenoHintListWidget::paintEvent(QPaintEvent* event)
{
    QPainter painter(this);
    painter.fillRect(rect(), "#22252C");
    painter.setPen(Qt::black);
    painter.drawRect(QRect(0, 0, width() - 1, height() - 1));
}

ZenoFuncDescriptionLabel::ZenoFuncDescriptionLabel(ZenoPropPanel* panel)
    : QWidget(panel)
    , m_currentFunc("")
    , m_parentPropanel(panel)
{
    setWindowFlags(windowFlags() | Qt::WindowStaysOnTopHint);
    //setMinimumSize({ 100, 50 });
    m_label = new QLabel(this);
    m_label->setStyleSheet("QLabel{ font: 12pt Microsoft Sans Serif; color: rgb(160, 178, 194)}");
    
    QVBoxLayout* layout = new QVBoxLayout(this);
    layout->setContentsMargins(10, 10, 10, 10);
    layout->addWidget(m_label);
    setLayout(layout);
    qApp->installEventFilter(this);

    hide();
}

void ZenoFuncDescriptionLabel::setDesc(zeno::FUNC_INFO func, int pos)
{
    std::string txtToSet = func.rettype + " " + func.name + " " + "(";
    for (int i = 0; i < func.args.size(); i++) {
        std::string arg_content = func.args[i].type + " " + func.args[i].name;
        if (i == pos) {
            txtToSet += "<b>" + arg_content + "</b> ";
        }
        else {
            txtToSet += arg_content + " ";
        }
    }
    txtToSet += ")";

    txtToSet = "<p>" + txtToSet + "</p>";
    txtToSet += "<p>" + func.tip + "</p>";
    txtToSet = "<html><head><style> p { margin: 10; } </style></head><body><div>" + txtToSet + "</div></body></html>";
    m_label->setText(QString::fromStdString(txtToSet));
    m_label->setFont(QApplication::font());
    adjustSize();
}

void ZenoFuncDescriptionLabel::setCurrentFuncName(std::string funcName)
{
    m_currentFunc = funcName;
}

std::string ZenoFuncDescriptionLabel::getCurrentFuncName()
{
    return m_currentFunc;
}

QPoint ZenoFuncDescriptionLabel::calculateNewPos(QWidget* widgetToFollow, const QString& txt)
{
    QFontMetrics metrics(widgetToFollow->font());
    if (QWidget* m_hintlistParent = qobject_cast<QWidget*>(this->parent())) {
        QPoint newpos = widgetToFollow->mapTo(m_hintlistParent, QPoint(0, 0));
        int parentwidth = m_hintlistParent->width();
        int txtwidth = metrics.width(txt);
        if (parentwidth < newpos.x() + txtwidth + this->width()) {
            newpos.setX(parentwidth - this->width());
        }
        else {
            newpos.setX(newpos.x() + txtwidth);
        }
        int parentheight = m_hintlistParent->height();
        if (parentheight < newpos.y() + widgetToFollow->height() + this->height()) {
            newpos.setY(newpos.y() - this->height());
        }
        else {
            newpos.setY(newpos.y() + widgetToFollow->height());
        }
        return newpos;
    }
    return { 0,0 };
}

void ZenoFuncDescriptionLabel::updateParent()
{
    auto mainwin = zenoApp->getMainWindow();
    ZASSERT_EXIT(mainwin);
    if (mainwin->propPanelIsFloating(m_parentPropanel)) {
        setParent(m_parentPropanel);
    }
    else {
        setParent(mainwin);
    }
}

bool ZenoFuncDescriptionLabel::eventFilter(QObject* watched, QEvent* event)
{
    if (this->isVisible())
    {
        if (event->type() == QEvent::MouseButtonPress)
        {
            if (QMouseEvent* e = static_cast<QMouseEvent*>(event))
            {
                if (QWidget* wid = qobject_cast<QWidget*>(watched)) //点击区域不在内部则hide
                {
                    const QPoint& globalPos = wid->mapToGlobal(e->pos());
                    const QPoint& lefttop = mapToGlobal(QPoint(0, 0));
                    const QPoint& rightbottom = mapToGlobal(QPoint(width(), height()));
                    if (globalPos.x() < lefttop.x() || globalPos.x() > rightbottom.x() || globalPos.y() < lefttop.y() || globalPos.y() > rightbottom.y())
                    {
                        hide();
                    }
                }
            }
        }
    }
    return QWidget::eventFilter(watched, event);
}

void ZenoFuncDescriptionLabel::paintEvent(QPaintEvent* event)
{
    QPainter painter(this);
    painter.fillRect(rect(), "#22252C");
    painter.setPen(Qt::black);
    painter.drawRect(QRect(0, 0, width() - 1, height() - 1));
}
