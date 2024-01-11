#include "zenonewmenu.h"
#include "model/graphsmanager.h"
#include "zenoapplication.h"
#include "fuzzy_search.h"
#include "nodeeditor/gv/zenoparamwidget.h"
#include "searchview.h"
#include "util/uihelper.h"


ZenoNewnodeMenu::ZenoNewnodeMenu(const QModelIndex& subgIdx, const NODE_CATES& cates, const QPointF& scenePos, QWidget* parent)
    : QMenu(parent)
    , m_preSearchMode(false)
    , m_cates(cates)
    , m_subgIdx(subgIdx)
    , m_scenePos(scenePos)
    , m_searchEdit(nullptr)
    , m_wactSearchEdit(nullptr)
{
    QPalette palette;
    palette.setColor(QPalette::Base, QColor(25, 29, 33));
    QColor clr = QColor(255, 255, 255);
    palette.setColor(QPalette::Text, clr);

    QFont font = zenoApp->font();
    // init search edit
    m_searchEdit = new ZenoGvLineEdit(this);
    m_searchEdit->setAutoFillBackground(false);
    m_searchEdit->setTextMargins(QMargins(8, 0, 0, 0));
    m_searchEdit->setPalette(palette);
    m_searchEdit->setFont(font);
    m_searchEdit->installEventFilter(this);
    // init search view
    m_searchView = new SearchResultWidget(this);
    m_searchView->setPalette(palette);
    m_searchView->setFont(font);
    m_searchView->installEventFilter(this);
    // init widget action
    m_wactSearchEdit = new QWidgetAction(this);
    m_wactSearchView = new QWidgetAction(this);
    m_wactSearchEdit->setDefaultWidget(m_searchEdit);
    m_wactSearchView->setDefaultWidget(m_searchView);

    addAction(m_wactSearchEdit);

    GraphsTreeModel* pModel = zenoApp->graphsManager()->currentModel();
    m_cateActions = getCategoryActions(m_subgIdx, m_scenePos);
    addActions(m_cateActions);

    // init [node, cate] map, [node...] list
    for (const NODE_CATE& cate : cates) {
        for (const QString& name : cate.nodes) {
            m_nodeToCate[name] = cate.name;
            m_condidates.push_back(name);
        }
    }
    connect(m_searchEdit, SIGNAL(textChanged(const QString&)), this, SLOT(onTextChanged(const QString&)));
    connect(m_searchView, &SearchResultWidget::clicked, this, [this](SearchResultItem* item) {
        GraphsTreeModel* pModel = zenoApp->graphsManager()->currentModel();
        if (!pModel) return;

        QString name = item->result();
        UiHelper::createNewNode(m_subgIdx, name, m_scenePos);
        this->close();
    });
}

ZenoNewnodeMenu::~ZenoNewnodeMenu()
{
}

void ZenoNewnodeMenu::setEditorFocus()
{
    m_searchEdit->setFocus();
}

bool ZenoNewnodeMenu::eventFilter(QObject* watched, QEvent* event)
{
    if (event->type() == QEvent::KeyPress)
    {
        QKeyEvent* pKeyEvent = static_cast<QKeyEvent*>(event);
        // when not focus m_searchEdit, still handle key input
        if (QMenu* pMenu = qobject_cast<QMenu*>(watched); pMenu || watched == m_searchView)
        {
            int ch = pKeyEvent->key();

            QChar c(ch);
            if (c.isLetterOrNumber())
            {
                QString text = m_searchEdit->text();
                text.append(c);
                m_searchEdit->setText(text);
                if (pMenu)
                    pMenu->hide();
                return true;
            }
        }
        else if (watched == m_searchEdit && pKeyEvent->key() == Qt::Key_Down)
        {
            focusNextPrevChild(true);
            if (m_preSearchMode)
            {
                m_searchView->setFocus();
                m_searchView->setCurrentRow(m_searchView->currentRow() + 1);
                return true;
            }
        }
        else if (watched == m_searchEdit && (pKeyEvent->key() == Qt::Key_Return || pKeyEvent->key() == Qt::Key_Enter))
        {
            if (m_searchView->isVisible() && m_searchView->count() > 0)
            {
                emit m_searchView->pressed(m_searchView->currentIndex());
            }
        }
    }
    else if (watched == m_searchEdit && event->type() == QEvent::Show)
    {
        m_searchEdit->activateWindow();
    }
    return QMenu::eventFilter(watched, event);
}

void ZenoNewnodeMenu::onTextChanged(const QString& text)
{
    bool searchMode = !text.isEmpty();
    // category action -> search result action
    if (searchMode && !m_preSearchMode) {
        for (auto act : m_cateActions) {
            removeAction(act);
        }
        addAction(m_wactSearchView);
    }
    // search result action -> category action
    else if (!searchMode && m_preSearchMode) {
        removeAction(m_wactSearchView);
        addActions(m_cateActions);
    }
    m_preSearchMode = searchMode;

    if (searchMode) {
        updateSearchView(text);
        m_searchView->moveToTop();
    }

    setEditorFocus();
}

QList<QAction*> ZenoNewnodeMenu::getCategoryActions(QModelIndex subgIdx, QPointF scenePos)
{
    if (!subgIdx.isValid())
        return QList<QAction*>();

    QList<QAction*> acts;
    int nodesNum = 0;
    if (m_cates.isEmpty())
    {
        QAction* pAction = new QAction("ERROR: no descriptors loaded!");
        pAction->setEnabled(false);
        acts.push_back(pAction);
        return acts;
    }

    for (const NODE_CATE& cate : m_cates)
    {
        QAction* pAction = new QAction(cate.name, this);
        QMenu* pChildMenu = new QMenu(this);
        pChildMenu->setToolTipsVisible(true);
        for (const QString& name : cate.nodes)
        {
            QAction* pChildAction = pChildMenu->addAction(name);
            //todo: tooltip
            connect(pChildAction, &QAction::triggered, [=]() {
                UiHelper::createNewNode(subgIdx, name, scenePos);
            });
        }
        pAction->setMenu(pChildMenu);
        pChildMenu->installEventFilter(this);
        acts.push_back(pAction);
    }
    return acts;
}

void ZenoNewnodeMenu::updateSearchView(const QString& filter)
{
    auto searchResult = fuzzy_search(filter, m_condidates);
    m_searchView->resizeCount(searchResult.size());

    // widgetaction does not update all the time, force to resize
    auto viewSize = m_searchView->sizeHint();
    auto editSize = m_searchEdit->sizeHint();
    auto height = viewSize.height() + editSize.height();
    auto width = std::max(viewSize.width(), editSize.width());
    this->resize(width, height);

    int deprecatedIndex = searchResult.size();
    for (int i = 0; i < deprecatedIndex;) {
        auto& [name, matchIndices] = searchResult[i];
        const auto& category = m_nodeToCate[name];
        if (category == "deprecated") {
            deprecatedIndex--;
            m_searchView->setResult(deprecatedIndex, name, matchIndices, category);
            std::swap(searchResult[i], searchResult[deprecatedIndex]);
        }
        else {
            m_searchView->setResult(i, name, matchIndices, category);
            ++i;
        }
    }
}
