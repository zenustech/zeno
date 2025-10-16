#include "zcodeeditor.h"
#include <QGLSLCompleter>
#include <QLuaCompleter>
#include "ZPythonCompleter.h"
#include <QZfxHighlighter>
#include <QGLSLHighlighter>
#include <QXMLHighlighter>
#include <QJSONHighlighter>
#include <QLuaHighlighter>
#include <QPythonHighlighter>
#include <QSyntaxStyle>
#include <zeno/core/data.h>
#include <zeno/core/Session.h>
#include "zenoapplication.h"
#include "zenomainwindow.h"
#include "style/zenostyle.h"
#include "util/uihelper.h"
#include <zeno/formula/formula.h>


ZCodeEditor::ZCodeEditor(const QString& text, QWidget *parent)
    : QCodeEditor(parent), m_zfxHighLighter(new QZfxHighlighter), m_descLabel(nullptr), m_hintlist(nullptr), m_nodeIdx(QModelIndex())
{
    setCompleter(new ZPythonCompleter(this));
    setHighlighter(m_zfxHighLighter);
    setText(text);
    initUI();

    connect(this, &QTextEdit::cursorPositionChanged, this, &ZCodeEditor::slt_showFuncDesc);
    //connect(this, &QTextEdit::textChanged, [this]() {
    //    int currDocHeight = this->document()->size().height();
    //    if (minimumHeight < currDocHeight && currDocHeight < ZenoStyle::dpiScaled(680)) {
    //        setFixedHeight(document()->size().height());
    //    }
    //});
}

void ZCodeEditor::setHintListWidget(ZenoHintListWidget* hintlist, ZenoFuncDescriptionLabel* descLabel)
{
    m_hintlist = hintlist;
    m_descLabel = descLabel;
}

void ZCodeEditor::setNodeIndex(QModelIndex nodeIdx)
{
    m_nodeIdx = nodeIdx;
}

void ZCodeEditor::focusInEvent(QFocusEvent* e)
{
    QCodeEditor::focusInEvent(e);
}

void ZCodeEditor::focusOutEvent(QFocusEvent* e)
{
    QCodeEditor::focusOutEvent(e);
    Qt::FocusReason reason = e->reason();
    if (reason != Qt::ActiveWindowFocusReason)
        emit editFinished(toPlainText());

    //setFixedHeight(minimumHeight);
}

void ZCodeEditor::keyPressEvent(QKeyEvent* event)
{
    if (m_hintlist && hasFocus())
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
                disconnect(m_hintlist, &ZenoHintListWidget::hintSelected, this, &ZCodeEditor::sltHintSelected);
                disconnect(m_hintlist, &ZenoHintListWidget::escPressedHide, this, &ZCodeEditor::sltSetFocus);
                disconnect(m_hintlist, &ZenoHintListWidget::resizeFinished, this, &ZCodeEditor::sltSetFocus);
                event->accept();
                return;
            }
            else if (event->key() == Qt::Key_Return || event->key() == Qt::Key_Enter)
            {
                if (m_hintlist->isVisible())
                {
                    hintSelectedSetText(m_hintlist->getCurrentText());
                    event->accept();
                    QTimer::singleShot(0, [this]() {
                        m_hintlist->hide();
                    });
                    return;
                }
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
                this->clearFocus();
            }
        }
    }
    QCodeEditor::keyPressEvent(event);
}

void ZCodeEditor::slt_showFuncDesc()
{
    if (!m_descLabel)
        return;

    QTextCursor cursor = textCursor();
    int positionInLine = cursor.positionInBlock();
    QString currLine = cursor.block().text().left(positionInLine);
    QRegularExpression functionPattern = (QRegularExpression(R"(\b([_a-zA-Z][_a-zA-Z0-9]*\s+)?((?:[_a-zA-Z][_a-zA-Z0-9]*\s*::\s*)*[_a-zA-Z][_a-zA-Z0-9]*\s*\([^)]*))"));
    auto matchIterator = functionPattern.globalMatch(currLine);

    QRegularExpression hintPattern = (QRegularExpression(R"(([_a-zA-Z][_a-zA-Z0-9]*)$)"));
    auto matchHintIterator = hintPattern.globalMatch(currLine);

    QString txt, nodePath;
    if (matchIterator.hasNext()){
        while (matchIterator.hasNext())
        {
            auto match = matchIterator.next();
            if (match.capturedStart(2) + match.capturedLength(2) == positionInLine) {
                txt = currLine.mid(match.capturedStart(2), match.capturedLength(2));
                nodePath = m_nodeIdx.data(QtRole::ROLE_OBJPATH).toString();
            }
        }
    } else {
        while (matchHintIterator.hasNext())
        {
            auto match = matchHintIterator.next();
            if (match.capturedStart(1) + match.capturedLength(1) == positionInLine) {
                txt = currLine.mid(match.capturedStart(1), match.capturedLength(1));
                nodePath = m_nodeIdx.data(QtRole::ROLE_OBJPATH).toString();
            }
        }
    }

    if (!txt.isEmpty() && !nodePath.isEmpty()) {
        zeno::Formula fmla(txt.toStdString(), nodePath.toStdString());

        //函数说明
        int ret = fmla.parse();
        //fmla.printSyntaxTree();
        if (ret == 0 || fmla.getASTResult())
        {
            auto getPos = [&]() -> QPoint {
                auto cursRect = cursorRect();
                auto globalpos = this->mapTo(zenoApp->getMainWindow(), { cursRect.x() + cursRect.width() + qCeil(ZenoStyle::dpiScaled(30)), cursRect.y() + cursRect.height() });
                if (zenoApp->getMainWindow()->width() < globalpos.x() + m_descLabel->width()) {
                    globalpos.setX(zenoApp->getMainWindow()->width() - m_descLabel->width());
                }
                if (zenoApp->getMainWindow()->height() < globalpos.y() + m_descLabel->height()) {
                    globalpos.setY(zenoApp->getMainWindow()->height() - m_descLabel->height());
                }
                return globalpos;
            };
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
                        connect(m_hintlist, &ZenoHintListWidget::hintSelected, this, &ZCodeEditor::sltHintSelected, Qt::UniqueConnection);
                        connect(m_hintlist, &ZenoHintListWidget::escPressedHide, this, &ZCodeEditor::sltSetFocus, Qt::UniqueConnection);
                        connect(m_hintlist, &ZenoHintListWidget::resizeFinished, this, &ZCodeEditor::sltSetFocus, Qt::UniqueConnection);
                        m_hintlist->updateParent();
                        m_hintlist->show();
                        if (m_descLabel->isVisible()) {
                            m_descLabel->hide();
                        }
                    }
                    m_hintlist->move(getPos());
                    //m_hintlist->move(m_hintlist->calculateNewPos(this, txt));
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
                    m_descLabel->move(getPos());
                    //m_descLabel->move(m_descLabel->calculateNewPos(this, txt));
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
    } else {
        m_hintlist->hide();
        m_descLabel->hide();
    }
}

void ZCodeEditor::sltHintSelected(QString itemSelected)
{
    hintSelectedSetText(itemSelected);
    setFocus();
    if (m_hintlist)
    {
        disconnect(m_hintlist, &ZenoHintListWidget::hintSelected, this, &ZCodeEditor::sltHintSelected);
        disconnect(m_hintlist, &ZenoHintListWidget::escPressedHide, this, &ZCodeEditor::sltSetFocus);
        disconnect(m_hintlist, &ZenoHintListWidget::resizeFinished, this, &ZCodeEditor::sltSetFocus);
    }
}

void ZCodeEditor::sltSetFocus()
{
    setFocus();
}

void ZCodeEditor::initUI()
{
    setSyntaxStyle(loadStyle(":/stylesheet/drakula.xml"));
    setWordWrapMode(QTextOption::WordWrap);
    setAutoIndentation(true);
    setTabReplace(true);
    setTabReplaceSize(4);

    setLineWrapMode(QTextEdit::WidgetWidth);

    ////初始化时显示所有内容
    //QFontMetrics fontMetrics(this->font());
    //int currDocHeight = this->document()->blockCount() * fontMetrics.height();
    //if (currDocHeight < minimumHeight) {
    //    setFixedHeight(minimumHeight);
    //}
    //else if (minimumHeight < currDocHeight && currDocHeight < ZenoStyle::dpiScaled(680)) {
    //    setFixedHeight(currDocHeight + ZenoStyle::dpiScaled(10));
    //}
    //else {
    //    setFixedHeight(ZenoStyle::dpiScaled(680));
    //}
    ////setMinimumHeight(minimumHeight);
}

QSyntaxStyle* ZCodeEditor::loadStyle(const QString& path)
{
    QFile fl(path);

    if (!fl.open(QIODevice::ReadOnly))
    {
        return QSyntaxStyle::defaultStyle();
    }

    auto style = new QSyntaxStyle(this);

    if (!style->load(fl.readAll()))
    {
        delete style;
        return QSyntaxStyle::defaultStyle();
    }

    return style;
}

void ZCodeEditor::hintSelectedSetText(QString text)
{
    //BlockSignalScope scope(this);
    QTextCursor cursor = textCursor();
    int newPos = cursor.position() - m_firstCandidateWord.size() + text.size();
    //QString txt = this->text();
    QString txt = this->toPlainText();
    txt.replace(cursor.position() - m_firstCandidateWord.size(), m_firstCandidateWord.size(), text);
    setText(txt);
    
    cursor.setPosition(newPos);
    setTextCursor(cursor);
}
