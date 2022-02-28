#include "framework.h"
#include "layerwidget.h"
#include "layertreeitemdelegate.h"


LayerTreeitemDelegate::LayerTreeitemDelegate(QWidget* parent)
    : QStyledItemDelegate(parent), m_treeview(nullptr)
{
    m_treeview = qobject_cast<LayerTreeView*>(parent);
}

QRect LayerTreeitemDelegate::getLockRect(const QStyleOptionViewItem *option) const
{
    int button_rightmargin = 10;
    int button_button = 12;
    int icon_sz = 32;
    int x = option->rect.right() - button_rightmargin - button_button - icon_sz * 2;
    int yoffset = 3;
    return QRect(x, option->rect.y() + yoffset, icon_sz, icon_sz);
}

QRect LayerTreeitemDelegate::getViewRect(const QStyleOptionViewItem *option) const
{
    int button_rightmargin = 10;
    int button_button = 12;
    int icon_sz = 32;
    int x = option->rect.right() - button_rightmargin - icon_sz;
    int yoffset = 3;
    return QRect(x, option->rect.y() + yoffset, icon_sz, icon_sz);
}

bool LayerTreeitemDelegate::editorEvent(QEvent *event, QAbstractItemModel *model, const QStyleOptionViewItem &option, const QModelIndex& index)
{
    if (event->type() == QEvent::MouseButtonPress)
    {
        QMouseEvent *me = static_cast<QMouseEvent *>(event);
        if (me->button() == Qt::RightButton)
        {
            
            QString id = index.data(NODEID_ROLE).toString();
            if (id != HEADER_ID && id != BODY_ID)
            {
                QMenu *menu = new QMenu(m_treeview);
                if (index.parent().isValid() && !index.parent().parent().isValid())
                {
                    //Component
                    QStandardItemModel *pModel = qobject_cast<QStandardItemModel *>(model);

                    QAction *pAddImage = new QAction("Add Image");
                    connect(pAddImage, &QAction::triggered, [=]() {

                        QString original = QFileDialog::getOpenFileName(m_treeview, tr("Select an image"),
                                                   ".", "JPEG (*.jpg *jpeg)\n"
                                                   "GIF (*.gif)\n"
                                                   "PNG (*.png)\n"
                                                   "Bitmap Files (*.bmp)\n");

                        if (original.isEmpty())
                            return;

                        QFileInfo f(original);
                        QString fn = f.fileName();

                        //QImage image = QImageReader(original).read();

                        QStandardItem *pItem = pModel->itemFromIndex(index);
                        QStandardItem *pNewChild = new QStandardItem(QIcon(), fn);

                        pNewChild->setData(original, NODEID_ROLE);
                        pNewChild->setData(original, NODEPATH_ROLE);
                        pNewChild->setData(false, NODELOCK_ROLE);
                        pNewChild->setData(true, NODEVISIBLE_ROLE);
                        pNewChild->setEditable(false);

                        pItem->appendRow(pNewChild);
                    });
                    menu->addAction(pAddImage);

                    QAction *pSetImage = new QAction("Set Image");
                    connect(pSetImage, &QAction::triggered, [=]() {
                        QString original = QFileDialog::getOpenFileName(m_treeview, tr("Select an image"),
                                                                        ".", "JPEG (*.jpg *jpeg)\n"
                                                                             "GIF (*.gif)\n"
                                                                             "PNG (*.png)\n"
                                                                             "Bitmap Files (*.bmp)\n");

                        if (original.isEmpty())
                            return;

                        QFileInfo f(original);
                        QString fn = f.fileName();

                        QStandardItem *pItem = pModel->itemFromIndex(index);
                        pItem->setData(original, NODEPATH_ROLE);   
                        pItem->setEditable(false);
                    });
                    menu->addAction(pSetImage);
                    menu->addAction("New text");
                } 
                else {
                    //Element

                }
            }
        }
    }
    if (event->type() == QEvent::MouseButtonRelease) {
        QMouseEvent *me = static_cast<QMouseEvent *>(event);
        QPoint pos = me->pos();
        QRect rcLock = getLockRect(&option);
        QRect rcView = getViewRect(&option);
        if (rcLock.contains(pos)) {
            bool bLock = index.data(NODELOCK_ROLE).toBool();
            model->setData(index, !bLock, NODELOCK_ROLE);
        }
        if (rcView.contains(pos)) {
            bool bVisible = index.data(NODEVISIBLE_ROLE).toBool();
            model->setData(index, !bVisible, NODEVISIBLE_ROLE);
        }
    }
    return _base::editorEvent(event, model, option, index);
}

void LayerTreeitemDelegate::paint(QPainter* painter, const QStyleOptionViewItem& option, const QModelIndex& index) const
{
    QStyleOptionViewItem opt = option;
    initStyleOption(&opt, index);

    QRect rc = option.rect;

    //draw icon
    int icon_xmargin = 5;
    int icon_sz = 32;
    int icon_ymargin = (rc.height() - icon_sz) / 2;
    int icon2text_xoffset = 5;
    int button_rightmargin = 10;
    int button_button = 12;
    int text_yoffset = 12;

    QColor bgColor, borderColor;
    if (opt.state & QStyle::State_Selected)
    {
        bgColor = QColor(193, 222, 236);
    }
    else if (opt.state & QStyle::State_MouseOver)
    {
        bgColor = QColor(239, 248, 254);
    }
    else
    {
        bgColor = QColor(255, 255, 255);
    }

    // draw the background
    QRect rcBg(QPoint(0, rc.top()), QPoint(rc.right(), rc.bottom()));
    painter->fillRect(rc, bgColor);

    if (!option.icon.isNull())
    {
        QRect iconRect(opt.rect.x() + icon_xmargin, opt.rect.y() + icon_ymargin, icon_sz, icon_sz);
        QIcon::State state = opt.state & QStyle::State_Open ? QIcon::On : QIcon::Off;
        opt.icon.paint(painter, iconRect, opt.decorationAlignment, QIcon::Normal, state);
    }

    //draw text
    QFont font("Microsoft YaHei", 9);
    QFontMetricsF fontMetrics(font);
    int w = fontMetrics.horizontalAdvance(opt.text);
    int h = fontMetrics.height();
    int x = opt.rect.x() + icon_xmargin + icon_sz + icon2text_xoffset;
    QRect textRect(x, opt.rect.y(), w, opt.rect.height());
    if (!opt.text.isEmpty())
    {
        painter->setPen(QColor(0, 0, 0));
        painter->setFont(font);
        painter->drawText(textRect, Qt::AlignVCenter, opt.text);
    }

    //draw button
    {
        QString id = index.data(NODEID_ROLE).toString();
        if (id == HEADER_ID || id == BODY_ID) {
            return;
        }

        QIcon icon;
        icon.addFile(":/icons/locked.svg", QSize(), QIcon::Normal, QIcon::On);
        icon.addFile(":/icons/locked_off.svg", QSize(), QIcon::Normal, QIcon::Off);
        bool bLocked = index.data(NODELOCK_ROLE).toBool();
        QIcon::State state = bLocked ? QIcon::On : QIcon::Off;
        QRect rcLock = getLockRect(&option);
        icon.paint(painter, rcLock, opt.decorationAlignment, QIcon::Normal, state);

        icon = QIcon();
        icon.addFile(":/icons/eye.svg", QSize(), QIcon::Normal, QIcon::On);
        icon.addFile(":/icons/eye_off.svg", QSize(), QIcon::Normal, QIcon::Off);

        bool bVisible = index.data(NODEVISIBLE_ROLE).toBool();
        state = bVisible ? QIcon::On : QIcon::Off;
        QRect rcView = getViewRect(&option);
        icon.paint(painter, rcView, opt.decorationAlignment, QIcon::Normal, state);
    }
}

QSize LayerTreeitemDelegate::sizeHint(const QStyleOptionViewItem& option, const QModelIndex& index) const
{
    int w = ((QWidget*)parent())->width();
    return QSize(w, 36);
}

void LayerTreeitemDelegate::initStyleOption(QStyleOptionViewItem* option, const QModelIndex& index) const
{
    QStyledItemDelegate::initStyleOption(option, index);
}