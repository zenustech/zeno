#include "zenosearchbar.h"
#include "../comctrl/ziconbutton.h"
#include "../model/subgraphmodel.h"
#include "../model/modelrole.h"


ZenoSearchBar::ZenoSearchBar(SubGraphModel *model, QWidget *parentWidget)
    : QWidget(parentWidget)
    , m_idx(0)
    , m_model(model)
{
    QVBoxLayout *pMainLayout = new QVBoxLayout;

    setWindowFlag(Qt::SubWindow);
    setWindowFlag(Qt::FramelessWindowHint);

    m_pLineEdit = new QLineEdit;
    m_pLineEdit->setFixedWidth(200);
    ZIconButton *pCloseBtn = new ZIconButton(QIcon(":/icons/closebtn.svg"), QSize(20, 20),
                                                   QColor(61, 61, 61), QColor(66, 66, 66));
    ZIconButton *pSearchBackward = new ZIconButton(QIcon(":/icons/search_arrow_backward.svg"), QSize(20, 20),
                                                   QColor(61, 61, 61), QColor(66, 66, 66));
    ZIconButton *pSearchForward = new ZIconButton(QIcon(":/icons/search_arrow.svg"), QSize(20, 20),
                                               QColor(61, 61, 61), QColor(66, 66, 66));
    QHBoxLayout *pEditLayout = new QHBoxLayout;
    
    pEditLayout->addWidget(m_pLineEdit);
    pEditLayout->addWidget(pSearchBackward);
    pEditLayout->addWidget(pSearchForward);
    pEditLayout->addWidget(pCloseBtn);
    
    pMainLayout->addLayout(pEditLayout);

    QHBoxLayout* pOptionLayout = new QHBoxLayout;
    pOptionLayout->addStretch();
    QComboBox* pSearchElem = new QComboBox;
    pSearchElem->addItems({"Name", "Object", "Param"});

    QComboBox *pSearchRange = new QComboBox;
    pSearchRange->addItems({"Current SubGraph", "All Graphs"});
    pOptionLayout->addWidget(pSearchElem);
    pOptionLayout->addWidget(pSearchRange);

    pMainLayout->addLayout(pOptionLayout);

    setLayout(pMainLayout);

    connect(m_pLineEdit, SIGNAL(textChanged(const QString&)), this, SLOT(onSearchExec(const QString &)));
    connect(pSearchForward, SIGNAL(clicked()), this, SLOT(onSearchForward()));
    connect(pSearchBackward, SIGNAL(clicked()), this, SLOT(onSearchBackward()));
    connect(pCloseBtn, SIGNAL(clicked()), this, SLOT(close()));
}

SEARCH_RECORD ZenoSearchBar::_getRecord()
{
    QModelIndex idx = m_results[m_idx];
    const QString &nodeid = idx.data(ROLE_OBJID).toString();
    const QPointF &pos = idx.data(ROLE_OBJPOS).toPointF();
    return {nodeid, pos};
}

void ZenoSearchBar::onSearchExec(const QString& content)
{
    m_results = m_model->match(m_model->index(0, 0), ROLE_OBJNAME, content, -1, Qt::MatchContains);
    if (!m_results.isEmpty())
    {
        m_idx = 0;
        SEARCH_RECORD rec = _getRecord();
        emit searchReached(rec);
    }
}

void ZenoSearchBar::onSearchForward()
{
    m_idx = qMin(m_idx + 1, m_results.size() - 1);
    if (!m_results.isEmpty() && m_idx < m_results.size())
    {
        SEARCH_RECORD rec = _getRecord();
        emit searchReached(rec);
    }
}

void ZenoSearchBar::onSearchBackward()
{
    m_idx = qMax(0, m_idx - 1);
    if (!m_results.isEmpty() && m_idx < m_results.size())
    {
        SEARCH_RECORD rec = _getRecord();
        emit searchReached(rec);
    }
}
