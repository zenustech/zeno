#include "zsubnetlistitemdelegate.h"
#include "style/zenostyle.h"
#include "zenosubnetlistview.h"
#include <model/graphsmodel.h>


ZSubnetListItemDelegate::ZSubnetListItemDelegate(GraphsPlainModel* model, QObject* parent)
    : QStyledItemDelegate(parent)
    , m_model(model)
{
}

ZSubnetListItemDelegate::~ZSubnetListItemDelegate()
{
    m_model = nullptr;
}

// painting
void ZSubnetListItemDelegate::paint(QPainter* painter, const QStyleOptionViewItem& option, const QModelIndex& index) const
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
    int text_xmargin = 12;

    QColor bgColor, borderColor, textColor;
    if (opt.state & QStyle::State_Selected)
    {
        bgColor = QColor(44, 73, 98);
        borderColor = QColor(27, 145, 225);
        textColor = QColor(255, 255, 255);

        painter->fillRect(rc, bgColor);
        painter->setPen(QPen(borderColor));
        painter->drawRect(rc.adjusted(0, 0, -1, -1));
    }
    else if (opt.state & QStyle::State_MouseOver)
    {
        textColor = QColor(255, 255, 255);
    }
    else
    {
        textColor = QColor(134, 130, 128);
    }

    if (!option.icon.isNull())
    {
        QRect iconRect(opt.rect.x() + icon_xmargin, opt.rect.y() + icon_ymargin, icon_sz, icon_sz);
        QIcon::State state = opt.state & QStyle::State_Open ? QIcon::On : QIcon::Off;
        opt.icon.paint(painter, iconRect, opt.decorationAlignment, QIcon::Normal, state);
    }

    //draw text
    QFont font("HarmonyOS Sans", 11);
    font.setBold(false);
    QFontMetricsF fontMetrics(font);
    int w = fontMetrics.horizontalAdvance(opt.text);
    int h = fontMetrics.height();
    int x = text_xmargin;// opt.rect.x() + icon_xmargin + icon_sz + icon2text_xoffset;
    QRect textRect(x, opt.rect.y(), w, opt.rect.height());
    if (!opt.text.isEmpty())
    {
        painter->setPen(textColor);
        painter->setFont(font);
        painter->drawText(textRect, Qt::AlignVCenter, opt.text);
    }
}

QSize ZSubnetListItemDelegate::sizeHint(const QStyleOptionViewItem& option, const QModelIndex& index) const
{
    int width = option.fontMetrics.horizontalAdvance(option.text);
    QFont fnt = option.font;
    return ZenoStyle::dpiScaledSize(QSize(180, 30));
}

void ZSubnetListItemDelegate::initStyleOption(QStyleOptionViewItem* option, const QModelIndex& index) const
{
    QStyledItemDelegate::initStyleOption(option, index);

}

bool ZSubnetListItemDelegate::editorEvent(QEvent* event, QAbstractItemModel* model, const QStyleOptionViewItem& option, const QModelIndex& index)
{
    if (event->type() == QEvent::MouseButtonPress)
    {
        QMouseEvent* me = static_cast<QMouseEvent*>(event);
        if (me->button() == Qt::RightButton)
        {
            QMenu* menu = new QMenu(qobject_cast<ZenoSubnetListView*>(parent()));
            QAction* pCopySubnet = new QAction("Copy subnet");
            QAction* pPasteSubnet = new QAction("Paste subnet");
            QAction* pRename = new QAction("Rename");
            QAction* pDelete = new QAction("Delete");

            connect(pDelete, &QAction::triggered, this, [=]() {
                onDelete(index);
                });

            menu->addAction(pCopySubnet);
            menu->addAction(pPasteSubnet);
            menu->addSeparator();
            menu->addAction(pRename);
            menu->addAction(pDelete);
            menu->exec(QCursor::pos());
        }
    }
    return QStyledItemDelegate::editorEvent(event, model, option, index);
}

void ZSubnetListItemDelegate::onDelete(const QModelIndex& index)
{
    //todo
    //m_model->removeGraph(index.row());
}

QWidget* ZSubnetListItemDelegate::createEditor(QWidget* parent, const QStyleOptionViewItem& option, const QModelIndex& index) const
{
    return QStyledItemDelegate::createEditor(parent, option, index);
}

void ZSubnetListItemDelegate::setEditorData(QWidget* editor, const QModelIndex& index) const
{
    QStyledItemDelegate::setEditorData(editor, index);
}

void ZSubnetListItemDelegate::setModelData(QWidget* editor, QAbstractItemModel* model, const QModelIndex& index) const
{
    QStyledItemDelegate::setModelData(editor, model, index);
}

void ZSubnetListItemDelegate::updateEditorGeometry(QWidget* editor, const QStyleOptionViewItem& option, const  QModelIndex& index) const
{
    QStyledItemDelegate::updateEditorGeometry(editor, option, index);
}