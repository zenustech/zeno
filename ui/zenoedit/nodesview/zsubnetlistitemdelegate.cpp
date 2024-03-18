#include "zsubnetlistitemdelegate.h"
#include "style/zenostyle.h"
#include "zenosubnetlistview.h"
#include <zenomodel/include/graphsmanagment.h>
#include "zenoapplication.h"
#include <zenomodel/include/modelrole.h>
#include "util/log.h"
#include <zenoio/writer/zsgwriter.h>
#include "zenoapplication.h"
#include "zenomainwindow.h"
#include "settings/zenosettingsmanager.h"

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
        bgColor = QColor(59, 64, 73);
        painter->fillRect(rc, bgColor);
    }

    if (!opt.icon.isNull())
    {
        QRect iconRect(opt.rect.x() + icon_xmargin, opt.rect.y() + icon_ymargin, icon_sz, icon_sz);
        QIcon::State state = opt.state & QStyle::State_Open ? QIcon::On : QIcon::Off;
        opt.icon.paint(painter, iconRect, opt.decorationAlignment, QIcon::Normal, state);
    }

    //draw text
    QFont font = QApplication::font();
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

bool ZSubnetListItemDelegate::editorEvent(QEvent* event, QAbstractItemModel* model, const QStyleOptionViewItem& option, const QModelIndex& proxyIndex)
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
            QAction* pSave = new QAction(tr("Save Subgrah"));

            if (m_selectedIndexs.size() > 1) 
            {
                pRename->setEnabled(false);
                pSave->setEnabled(false);
            }
            connect(pDelete, &QAction::triggered, this, [=]() {
                onDelete();
             });

            QSortFilterProxyModel* pProxyModel = qobject_cast<QSortFilterProxyModel*>(model);
            ZASSERT_EXIT(pProxyModel, false);
            const QModelIndex& index = pProxyModel->mapToSource(proxyIndex);

            connect(pRename, &QAction::triggered, this, [=]() {
                onRename(index);
            });

            connect(pSave, &QAction::triggered, this, [=]() {
                onSaveSubgraph(index);
            });

            bool bForkLocked = index.data(ROLE_FORK_LOCKSTATUS).toBool();
            QAction* pForkLock = new QAction(bForkLocked ? tr("Fork unlocked") : tr("Fork locked"));
            connect(pForkLock, &QAction::triggered, this, [=]() {
                setForkLock(pProxyModel, !bForkLocked);
            });

            menu->addAction(pCopySubnet);
            menu->addAction(pPasteSubnet);
            menu->addSeparator();
            menu->addAction(pRename);
            menu->addAction(pDelete);
            menu->addAction(pSave);
            menu->addAction(pForkLock);
            bool bPreset = index.data(ROLE_SUBGRAPH_TYPE) == SUBGRAPH_PRESET;
            QAction* pPreset = new QAction(bPreset ? tr("Trans to Normal Subgrah") : tr("Trans to Preset Subgrah"));
            menu->addAction(pPreset);
            connect(pPreset, &QAction::triggered, this, [=]() {
                setSubgraphType(pProxyModel, bPreset);
            });

            menu->exec(QCursor::pos());
        }
    }
    return QStyledItemDelegate::editorEvent(event, model, option, proxyIndex);
}

void ZSubnetListItemDelegate::onDelete()
{
    int button = QMessageBox::question(nullptr, tr("Delete Subgraph"), tr("Do you want to delete the selected subgraphs"));
    if (button == QMessageBox::Yes) {
        QStringList nameList;
        for (const QModelIndex &idx : m_selectedIndexs) {
            QString subgName = idx.data(ROLE_OBJNAME).toString();
            if (subgName.compare("main", Qt::CaseInsensitive) == 0) {
                QMessageBox msg(QMessageBox::Warning, tr("Zeno"), tr("main graph is not allowed to be deleted"));
                msg.exec();
                continue;
            }
            nameList << subgName;
        }
        for (const QString& name : nameList)
        {
             m_model->removeSubGraph(name);
        }
    }
}

void ZSubnetListItemDelegate::onRename(const QModelIndex &index) 
{
    QString name = QInputDialog::getText(nullptr, tr("Rename"), tr("subgraph name:"), QLineEdit::Normal, index.data(ROLE_OBJNAME).toString());
    if (!name.isEmpty()) {
        m_model->setData(index, name, Qt::EditRole);
    }
}

void ZSubnetListItemDelegate::onSaveSubgraph(const QModelIndex& index)
{
    DlgInEventLoopScope;
    QString subgName = index.data(ROLE_OBJNAME).toString();
    QString path = QFileDialog::getSaveFileName(nullptr, "Path to Save", subgName, "Zeno Graph File(*.zsg);; All Files(*);;");
    if (!path.isEmpty()) {
        QModelIndexList lst;
        getSubgraphs(index, lst);
        QString strJson = ZsgWriter::getInstance().dumpSubgraphStr(m_model, lst);
        QFile file(path);
        zeno::log_debug("saving {} chars to file [{}]", strJson.size(), path.toStdString());
        if (!file.open(QIODevice::WriteOnly)) {
            qWarning() << Q_FUNC_INFO << "Failed to open" << strJson << file.errorString();
            zeno::log_error("Failed to open file for write: {} ({})", path.toStdString(), file.errorString().toStdString());
            return;
        }

        file.write(strJson.toUtf8());
        file.close();
        zeno::log_debug("saved subgraph {} successfully", subgName.toStdString());
    }
}

void ZSubnetListItemDelegate::getSubgraphs(const QModelIndex& subgIdx, QModelIndexList& subgraphs)
{
    int count = m_model->itemCount(subgIdx);
    for (int i = 0; i < count; i++)
    {
        const QModelIndex& childIdx = m_model->index(i, subgIdx);
        if (!childIdx.isValid())
            continue;
        const QString& subgName = childIdx.data(ROLE_OBJNAME).toString();
        const QModelIndex& modelIdx = m_model->index(subgName);
        if (modelIdx.isValid() && !subgraphs.contains(modelIdx))
        {
            getSubgraphs(modelIdx, subgraphs);
        }
    }
    subgraphs << subgIdx;
}

QWidget* ZSubnetListItemDelegate::createEditor(QWidget* parent, const QStyleOptionViewItem& option, const QModelIndex& index) const
{
    QLineEdit* pLineEdit = qobject_cast<QLineEdit*>(QStyledItemDelegate::createEditor(parent, option, index));
    ZASSERT_EXIT(pLineEdit, nullptr);
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
void ZSubnetListItemDelegate::setSelectedIndexs(const QModelIndexList &list) 
{
    m_selectedIndexs = list;
}

void ZSubnetListItemDelegate::setForkLock(QSortFilterProxyModel* pProxyModel, bool bLocked)
{
    QModelIndexList indexs;
    for (const QModelIndex& idx : m_selectedIndexs) {
        const QModelIndex& srcIdx = pProxyModel->mapToSource(idx);
        if (srcIdx.isValid())
            indexs << srcIdx;
    }
    for (const auto& idx : indexs)
    {
        m_model->setData(idx, bLocked, ROLE_FORK_LOCKSTATUS);
    }
}

void ZSubnetListItemDelegate::setSubgraphType(QSortFilterProxyModel* pProxyModel, bool bPreset)
{
    QModelIndexList indexs;
    for (const QModelIndex& idx : m_selectedIndexs) {
        const QModelIndex& srcIdx = pProxyModel->mapToSource(idx);
        if (srcIdx.isValid())
            indexs << srcIdx;
    }
    for (const auto& idx : indexs)
    {
        m_model->setData(idx, bPreset ? SUBGRAPH_NOR : SUBGRAPH_PRESET, ROLE_SUBGRAPH_TYPE);
    }
}


SubListSortProxyModel::SubListSortProxyModel(QObject* parent) : QSortFilterProxyModel(parent)
{
    connect(&ZenoSettingsManager::GetInstance(), &ZenoSettingsManager::valueChanged, this, [=](QString zsName) {
        if (zsName == zsSubgraphType)
        {
            invalidate();
        }
    });
}

bool SubListSortProxyModel::lessThan(const QModelIndex& source_left, const QModelIndex& source_right) const
{
    if (source_right.data().toString() == "main")
        return false;
    if (source_left.data().toString().compare(source_right.data().toString(), Qt::CaseInsensitive) < 0)
        return true;
    return false;
}

bool SubListSortProxyModel::filterAcceptsRow(int source_row, const QModelIndex& source_parent) const
{
    const QModelIndex& index = sourceModel()->index(source_row, 0, source_parent);
    if (!index.isValid())
        return false;
    int value = index.data(ROLE_SUBGRAPH_TYPE).toInt();
    int type = ZenoSettingsManager::GetInstance().getValue(zsSubgraphType).toInt();
    return type == value ? true : false;
}
