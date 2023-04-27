#include "zsubnetlistitemdelegate.h"
#include "style/zenostyle.h"
#include "zenosubnetlistview.h"
#include <zenomodel/include/graphsmanagment.h>
#include "zenoapplication.h"
#include <zenomodel/include/modelrole.h>
#include "util/log.h"


SubgEditValidator::SubgEditValidator(QObject* parent)
{
}

SubgEditValidator::~SubgEditValidator()
{
}

QValidator::State SubgEditValidator::validate(QString& input, int& pos) const
{
    if (input.isEmpty())
        return Intermediate;

    IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
    const QModelIndex& idx = pModel->index(input);
    if (idx.isValid())
        return Intermediate;

    return Acceptable;
}

void SubgEditValidator::fixup(QString& wtf) const
{

}


ZSubnetListItemDelegate::ZSubnetListItemDelegate(IGraphsModel* model, QObject* parent)
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
    int icon_xmargin = ZenoStyle::dpiScaled(20);
    int icon_sz = rc.height() * 0.8;
    int icon_ymargin = (rc.height() - icon_sz) / 2;
    int icon2text_xoffset = ZenoStyle::dpiScaled(7);
    int button_rightmargin = ZenoStyle::dpiScaled(10);
    int button_button = ZenoStyle::dpiScaled(12);
    int text_yoffset = ZenoStyle::dpiScaled(8);
    int text_xmargin = ZenoStyle::dpiScaled(12);

    QColor bgColor, borderColor, textColor;
    textColor = QColor("#C3D2DF");
    if (opt.state & QStyle::State_Selected)
    {
        bgColor = QColor(59, 64, 73);
        borderColor = QColor(27, 145, 225);

        painter->fillRect(rc, bgColor);
        //painter->setPen(QPen(borderColor));
        //painter->drawRect(rc.adjusted(0, 0, -1, -1));
    }
    else if (opt.state & QStyle::State_MouseOver)
    {
        //textColor = QColor(255, 255, 255);
    }

    if (!opt.icon.isNull())
    {
        QRect iconRect(opt.rect.x() + icon_xmargin, opt.rect.y() + icon_ymargin, icon_sz, icon_sz);
        QIcon::State state = opt.state & QStyle::State_Open ? QIcon::On : QIcon::Off;
        opt.icon.paint(painter, iconRect, opt.decorationAlignment, QIcon::Normal, state);
    }

    //draw text
    QFont font = zenoApp->font();
    font.setPointSize(10);
    font.setBold(false);
    QFontMetricsF fontMetrics(font);
    int w = fontMetrics.horizontalAdvance(opt.text);
    int h = fontMetrics.height();
    int x = opt.rect.x() + icon_xmargin + icon_sz + icon2text_xoffset;
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
    return ZenoStyle::dpiScaledSize(QSize(180, 35));
}

void ZSubnetListItemDelegate::initStyleOption(QStyleOptionViewItem* option, const QModelIndex& index) const
{
    QStyledItemDelegate::initStyleOption(option, index);

	if (option->text.compare("main", Qt::CaseInsensitive) == 0)
	{
		option->icon = QIcon(":/icons/subnet-main.svg");
	}
	else
	{
        option->icon = QIcon(":/icons/subnet-general.svg");
	}
}

bool ZSubnetListItemDelegate::editorEvent(QEvent* event, QAbstractItemModel* model, const QStyleOptionViewItem& option, const QModelIndex& index)
{
    if (event->type() == QEvent::MouseButtonPress)
    {
        QMouseEvent* me = static_cast<QMouseEvent*>(event);
        if (me->button() == Qt::RightButton)
        {
            QMenu* menu = new QMenu(qobject_cast<QWidget*>(parent()));
            QAction* pCopySubnet = new QAction(tr("Copy subnet"));
            QAction* pPasteSubnet = new QAction(tr("Paste subnet"));
            QAction* pRename = new QAction(tr("Rename"));
            QAction* pDelete = new QAction(tr("Delete"));

            connect(pDelete, &QAction::triggered, this, [=]() {
                onDelete(index);
                });

            connect(pRename, &QAction::triggered, this, [=]() {
                onRename(index);
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
    QString subgName = index.data(ROLE_OBJNAME).toString();
    if (subgName.compare("main", Qt::CaseInsensitive) == 0)
    {
        QMessageBox msg(QMessageBox::Warning, tr("Zeno"), tr("main graph is not allowed to be deleted"));
        msg.exec();
        return;
    }
    m_model->removeSubGraph(subgName);
}

void ZSubnetListItemDelegate::onRename(const QModelIndex &index) 
{
    QString name = QInputDialog::getText(nullptr, tr("Rename"), tr("subgraph name:"));
    if (!name.isEmpty()) {
        m_model->setData(index, name, Qt::EditRole);
    }
}

QWidget* ZSubnetListItemDelegate::createEditor(QWidget* parent, const QStyleOptionViewItem& option, const QModelIndex& index) const
{
    QLineEdit* pLineEdit = qobject_cast<QLineEdit*>(QStyledItemDelegate::createEditor(parent, option, index));
    ZASSERT_EXIT(pLineEdit, nullptr);
    SubgEditValidator* pValidator = new SubgEditValidator;
    pLineEdit->setValidator(pValidator);
    return pLineEdit;
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