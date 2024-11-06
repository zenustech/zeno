#include "ztextedit.h"
#include "panel/ZenoHintListWidget.h"
#include <zeno/formula/formula.h>
#include "util/apphelper.h"
#include "util/uihelper.h"


ZTextEdit::ZTextEdit(QWidget* parent)
    : QTextEdit(parent), m_realLineCount(0), m_bShowHintList(true), m_hintlist(nullptr), m_descLabel(nullptr)
{
    initUI();
}

ZTextEdit::ZTextEdit(const QString& text, QWidget* parent)
    : QTextEdit(text, parent), m_realLineCount(0)
{
    initUI();
}

void ZTextEdit::setNodeIdx(const QModelIndex& index) {
    m_index = index;
}

void ZTextEdit::initUI()
{
    setSizeAdjustPolicy(QAbstractScrollArea::AdjustToContents);
    QTextDocument *pTextDoc = document();
    connect(pTextDoc, &QTextDocument::contentsChanged, this, [=]() {
        QSize s(document()->size().toSize());
        updateGeometry();
        emit geometryUpdated();

        //判断实际显示的行数变化
        QFontMetrics metrics(font());
        int currline = qCeil(static_cast<double>(document()->size().height()) / metrics.lineSpacing());
        if (currline != m_realLineCount) {
            emit lineCountReallyChanged(m_realLineCount, currline);
            m_realLineCount = currline;
        }

        //显示提示列表
        if (m_hintlist && m_descLabel && hasFocus() && m_bShowHintList && m_index.isValid())
        {
            QString txt = toPlainText().left(textCursor().position());
            QString nodePath = m_index.data(ROLE_OBJPATH).toString();
            zeno::Formula fmla(txt.toStdString(), nodePath.toStdString());

            const QTextCursor& cursor = textCursor();
            const QTextBlock& block = cursor.block();
            QString lineText;
            if (QTextLayout* layout = block.layout()) {
                int cursorPosInBlock = cursor.position() - block.position();
                int visualLineStart = 0;
                for (int i = 0; i < layout->lineCount(); ++i) {
                    QTextLine line = layout->lineAt(i);
                    auto x = line.textStart();
                    auto xx = line.textStart() + line.textLength();
                    if (cursorPosInBlock >= line.textStart() && cursorPosInBlock <= line.textStart() + line.textLength()) {
                        visualLineStart = line.textStart();
                        break;
                    }
                }
                lineText = block.text().mid(visualLineStart, cursorPosInBlock - visualLineStart);
            }

            //函数说明
            int ret = fmla.parse();
            //fmla.printSyntaxTree();
            if (ret == 0 || fmla.getASTResult())
            {
                zeno::formula_tip_info recommandInfo = fmla.getRecommandTipInfo();
                if (recommandInfo.type == zeno::FMLA_TIP_FUNC_CANDIDATES ||
                    recommandInfo.type == zeno::FMLA_TIP_REFERENCE)
                {
                    QStringList items;
                    std::string candidateWord = recommandInfo.prefix;
                    for (auto& item : recommandInfo.func_candidats) {
                        items << QString::fromStdString(item);
                    }
                    for (auto& item : recommandInfo.ref_candidates) {
                        items << QString::fromStdString(item.nodename);
                    }
                    m_firstCandidateWord = QString::fromStdString(candidateWord);

                    if (items.size() == 0) {
                        if (m_hintlist->isVisible()) {
                            m_hintlist->hide();
                        }
                    }
                    else {
                        m_hintlist->setData(items);
                        if (!m_hintlist->isVisible())
                        {
                            connect(m_hintlist, &ZenoHintListWidget::hintSelected, this, &ZTextEdit::sltHintSelected, Qt::UniqueConnection);
                            connect(m_hintlist, &ZenoHintListWidget::escPressedHide, this, &ZTextEdit::sltSetFocus, Qt::UniqueConnection);
                            connect(m_hintlist, &ZenoHintListWidget::resizeFinished, this, &ZTextEdit::sltSetFocus, Qt::UniqueConnection);
                            m_hintlist->updateParent();
                            m_hintlist->show();
                            if (m_descLabel->isVisible()) {
                                m_descLabel->hide();
                            }
                        }
                        m_hintlist->move(m_hintlist->calculateNewPos(this, lineText));
                        m_hintlist->resetCurrentItem();
                    }
                }
                else if (recommandInfo.type == zeno::FMLA_TIP_FUNC_ARGS)
                {
                    m_hintlist->hide();
                    if (recommandInfo.func_args.func.name.empty()) {
                        m_descLabel->hide();
                    }
                    else {
                        int pos = recommandInfo.func_args.argidx;
                        m_descLabel->setDesc(recommandInfo.func_args.func, recommandInfo.func_args.argidx - 1);
                        if (!m_descLabel->isVisible()) {
                            m_descLabel->updateParent();
                            m_descLabel->show();
                        }
                        m_descLabel->move(m_descLabel->calculateNewPos(this, lineText));
                        m_descLabel->setCurrentFuncName(recommandInfo.func_args.func.name);
                    }
                }
                else if (recommandInfo.type == zeno::FMLA_NO_MATCH)
                {
                    m_hintlist->hide();
                    m_descLabel->hide();
                }
            }
            else if (m_descLabel->isVisible()) {
                m_descLabel->hide();
            }
        }
    });
}

QSize ZTextEdit::minimumSizeHint() const
{
    QSize minSz = QTextEdit::minimumSizeHint();
    return minSz;
}

QSize ZTextEdit::sizeHint() const
{
    QSize sz = QTextEdit::sizeHint();
    return sz;
}

QSize ZTextEdit::viewportSizeHint() const
{
    QSize sz = document()->size().toSize();
    return sz;
}

void ZTextEdit::setHintListWidget(ZenoHintListWidget* hintlist, ZenoFuncDescriptionLabel* descLabl)
{
    m_hintlist = hintlist;
    m_descLabel = descLabl;
}

void ZTextEdit::hintSelectedSetText(QString itemSelected)
{
    BlockSignalScope scope(this);
    QTextCursor cursor = textCursor();
    int newPos = cursor.position() - m_firstCandidateWord.size() + itemSelected.size();
    QString txt = this->toPlainText();
    txt.replace(cursor.position() - m_firstCandidateWord.size(), m_firstCandidateWord.size(), itemSelected);
    setText(txt);
    textCursor().setPosition(newPos);
    setTextCursor(cursor);
}

void ZTextEdit::sltHintSelected(QString itemSelected)
{
    hintSelectedSetText(itemSelected);
    setFocus();
    if (m_hintlist)
    {
        disconnect(m_hintlist, &ZenoHintListWidget::hintSelected, this, &ZTextEdit::sltHintSelected);
        disconnect(m_hintlist, &ZenoHintListWidget::escPressedHide, this, &ZTextEdit::sltSetFocus);
        disconnect(m_hintlist, &ZenoHintListWidget::resizeFinished, this, &ZTextEdit::sltSetFocus);
    }
}

void ZTextEdit::sltSetFocus()
{
    setFocus();
}

void ZTextEdit::keyPressEvent(QKeyEvent* event)
{
    if (m_hintlist && hasFocus() && m_bShowHintList)
    {
        if (m_hintlist->isVisible())
        {
            if (event->key() == Qt::Key_Down || event->key() == Qt::Key_Up) {
                bool bDown = event->key() == Qt::Key_Down;
                m_hintlist->onSwitchItemByKey(bDown);
                event->accept();
                return;
            }
            else if (event->key() == Qt::Key_Escape)
            {
                m_hintlist->hide();
                setFocus();
                disconnect(m_hintlist, &ZenoHintListWidget::hintSelected, this, &ZTextEdit::sltHintSelected);
                disconnect(m_hintlist, &ZenoHintListWidget::escPressedHide, this, &ZTextEdit::sltSetFocus);
                disconnect(m_hintlist, &ZenoHintListWidget::resizeFinished, this, &ZTextEdit::sltSetFocus);
                event->accept();
                return;
            }
            else if (event->key() == Qt::Key_Return || event->key() == Qt::Key_Enter)
            {
                hintSelectedSetText(m_hintlist->getCurrentText());
                event->accept();
                //不知道为什么调用m_hintlist->hide()无法隐藏，只能在singleshot里调用才有效
                QTimer::singleShot(0, [this]() {
                    m_hintlist->hide();
                });
                return;
            }
        }
        else if (m_descLabel && m_descLabel->isVisible())
        {
            if (event->key() == Qt::Key_Escape || event->key() == Qt::Key_Return || event->key() == Qt::Key_Enter) {
                m_descLabel->hide();
                setFocus();
                event->accept();
            }
        }
        else {
            if (event->key() == Qt::Key_Escape) {
                QFocusEvent* event = new QFocusEvent(QEvent::FocusOut);
                qApp->sendEvent(this, event);
            }
        }
    }
    QTextEdit::keyPressEvent(event);
}

void ZTextEdit::focusInEvent(QFocusEvent* e)
{
    QTextEdit::focusInEvent(e);
}

void ZTextEdit::focusOutEvent(QFocusEvent* e)
{
    if (m_hintlist && !m_hintlist->isVisible())
    {
        disconnect(m_hintlist, &ZenoHintListWidget::hintSelected, this, &ZTextEdit::sltHintSelected);
        disconnect(m_hintlist, &ZenoHintListWidget::escPressedHide, this, &ZTextEdit::sltSetFocus);
        disconnect(m_hintlist, &ZenoHintListWidget::resizeFinished, this, &ZTextEdit::sltSetFocus);
    }
    QTextEdit::focusOutEvent(e);
    emit editFinished();
}

void ZTextEdit::resizeEvent(QResizeEvent* event)
{
    QSize s(document()->size().toSize());
    QTextEdit::resizeEvent(event);
    updateGeometry();
    emit geometryUpdated();
}