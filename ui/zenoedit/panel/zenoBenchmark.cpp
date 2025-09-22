#include "zenoBenchmark.h"
#include "zenomainwindow.h"
#include "zenoapplication.h"
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QGroupBox>
#include <QCompleter>
#include <QPushButton>
#include <QHeaderView>
#include <algorithm>
#include <QStringListModel>
#include <QSplitter>
#include <zenoui/style/zenostyle.h>
#include "nodesview/zenographseditor.h"
#include "nodesys/zenosubgraphscene.h"
#include "nodesys/zenonode.h"

// NodeListModel 实现
NodeListModel::NodeListModel(QObject *parent) : QAbstractListModel(parent) {}

int NodeListModel::rowCount(const QModelIndex &parent) const {
    return parent.isValid() ? 0 : m_nodes.size();
}

QVariant NodeListModel::data(const QModelIndex &index, int role) const {
    if (!index.isValid() || index.row() >= m_nodes.size())
        return QVariant();
    
    if (role == Qt::DisplayRole || role == Qt::EditRole)
        return m_nodes.at(index.row());
    
    return QVariant();
}

void NodeListModel::addNode(const QString &node) {
    if (m_nodes.contains(node))
        return;
    
    beginInsertRows(QModelIndex(), m_nodes.size(), m_nodes.size());
    m_nodes.append(node);
    endInsertRows();
}

void NodeListModel::removeNode(int index) {
    if (index < 0 || index >= m_nodes.size())
        return;
    
    beginRemoveRows(QModelIndex(), index, index);
    m_nodes.removeAt(index);
    endRemoveRows();
}

void NodeListModel::clear() {
    beginResetModel();
    m_nodes.clear();
    endResetModel();
}

QStringList NodeListModel::getNodes() const {
    return m_nodes;
}

// BenchmarkTableModel 实现
BenchmarkTableModel::BenchmarkTableModel(QObject *parent) : QAbstractTableModel(parent) {}

int BenchmarkTableModel::rowCount(const QModelIndex &parent) const {
    return parent.isValid() ? 0 : m_data.size();
}

int BenchmarkTableModel::columnCount(const QModelIndex &parent) const {
    return parent.isValid() ? 0 : 6;
}

QVariant BenchmarkTableModel::data(const QModelIndex &index, int role) const {
    if (!index.isValid() || index.row() >= m_data.size() || index.column() >= 6)
        return QVariant();
    
    if (role == Qt::DisplayRole || role == Qt::EditRole) {
        auto item = m_data.at(index.row());
        switch (index.column()) {
        case 0: return item.avg;
        case 1: return item.min;
        case 2: return item.max;
        case 3: return item.total;
        case 4: return item.cnt;
        case 5: return item.tag;
        default: return QVariant();
        }
    }
    
    //if (role == Qt::TextAlignmentRole && index.column() < 5) {
    //    return Qt::AlignRight + Qt::AlignVCenter;
    //}
    
    return QVariant();
}

QVariant BenchmarkTableModel::headerData(int section, Qt::Orientation orientation, int role) const {
    if (role != Qt::DisplayRole)
        return QVariant();
    
    if (orientation == Qt::Horizontal) {
        switch (section) {
        case 0: return "avg";
        case 1: return "min";
        case 2: return "max";
        case 3: return "total";
        case 4: return "cnt";
        case 5: return "tag";
        default: return QVariant();
        }
    }
    
    return QVariant();
}

void BenchmarkTableModel::sort(int column, Qt::SortOrder order) {
    if (column < 0 || column >= 6)
        return;
    
    beginResetModel();
    
    std::sort(m_data.begin(), m_data.end(), [column, order](const BenchmarkData &a, const BenchmarkData &b) {
        bool lessThan = false;
        switch (column) {
        case 0: lessThan = a.avg < b.avg; break;
        case 1: lessThan = a.min < b.min; break;
        case 2: lessThan = a.max < b.max; break;
        case 3: lessThan = a.total < b.total; break;
        case 4: lessThan = a.cnt < b.cnt; break;
        case 5: lessThan = a.tag < b.tag; break;
        default: lessThan = false;
        }
        return order == Qt::AscendingOrder ? lessThan : !lessThan;
    });
    
    endResetModel();
}

void BenchmarkTableModel::clear() {
    beginResetModel();
    m_data.clear();
    endResetModel();
}

void BenchmarkTableModel::reset(const std::string& data) {
    beginResetModel();
    m_data.clear();
    
    QStringList lines = QString::fromStdString(data).split('\n', Qt::SkipEmptyParts);
    
    for (int i = 1; i < lines.size(); ++i) {
        QString line = lines[i].trimmed();
        if (line.isEmpty()) continue;
        
        QStringList columns = line.split('|', Qt::SkipEmptyParts);
        if (columns.size() < 6) continue; // 确保有6列数据
        
        // 去除每列的空格
        for (int j = 0; j < columns.size(); ++j) {
            columns[j] = columns[j].trimmed();
        }
        
        // 解析数值数据
        bool ok;
        int avg = columns[0].toInt(&ok);
        if (!ok) continue;
        
        int min = columns[1].toInt(&ok);
        if (!ok) continue;
        
        int max = columns[2].toInt(&ok);
        if (!ok) continue;
        
        int total = columns[3].toInt(&ok);
        if (!ok) continue;
        
        int cnt = columns[4].toInt(&ok);
        if (!ok) continue;
        
        QString tag = columns[5]; // 第6列是标签
        
        BenchmarkData benchmarkData;
        benchmarkData.avg = avg;
        benchmarkData.min = min;
        benchmarkData.max = max;
        benchmarkData.total = total;
        benchmarkData.cnt = cnt;
        benchmarkData.tag = tag;
        
        m_data.append(benchmarkData);
    }
    
    endResetModel();
}

// zenoBenchmark 实现
zenoBenchmark::zenoBenchmark(QWidget* parent) : QWidget(parent)
{
    // 创建模型
    m_nodeListModel = new NodeListModel(this);
    m_benchmarkTableModel = new BenchmarkTableModel(this);
    m_tableProxyModel = new QSortFilterProxyModel(this);
    m_tableProxyModel->setSourceModel(m_benchmarkTableModel);
    
    setupUI();
    setupConnections();
}

zenoBenchmark::~zenoBenchmark()
{
}

std::string zenoBenchmark::monitoredNodes()
{
    QStringList nodes = std::move(m_nodeListModel->getNodes());
    std::string result;
    for (const QString &node : nodes) {
        result += node.toStdString();
    }
    return result;
}

void zenoBenchmark::setBenchMarkData(std::string data)
{
    m_benchmarkTableModel->reset(data);
}

void zenoBenchmark::setupUI() {
    // 设置主布局，去掉边距和间距
    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(0, 0, 0, 0);  // 减少外边距

    QHBoxLayout* topLayout = new QHBoxLayout();

    // 过滤节点输入框
    m_filterCb = new QCheckBox(this);
    m_filterCb->setProperty("cssClass", "proppanel");
    m_filterCb->setText(tr("filter nodes"));
    m_filterCb->setMinimumWidth(1);

    m_serachLabel = new QLabel;
    m_serachLabel->setText(tr("search"));
    m_serachLabel->setProperty("cssClass", "proppanel");
    m_serachLabel->setMinimumWidth(1);
    m_searchEdit = new QLineEdit;
    m_searchEdit->setProperty("cssClass", "proppanel");
    m_searchEdit->setMinimumWidth(1);

    topLayout->addWidget(m_filterCb);
    topLayout->addStretch();
    topLayout->addWidget(m_serachLabel);
    topLayout->addWidget(m_searchEdit);
    
    // 创建分割器
    QSplitter *splitter = new QSplitter(Qt::Horizontal, this);
    splitter->setStyleSheet("QSplitter::handle {background-color: rgb(31,31,31); width: 2px;}");

    // 左边部分
    QWidget *leftWidget = new QWidget;
    leftWidget->setVisible(false);
    QVBoxLayout* leftLayout = new QVBoxLayout(leftWidget);
    leftLayout->setContentsMargins(0,2,1,2);  // 减少内边距
    leftLayout->setSpacing(2);  // 减少内部组件间距
    
    m_filterEdit = new QLineEdit;
    m_filterEdit->setProperty("cssClass", "proppanel");
    
    m_completer = new QCompleter(this);
    m_completer->setCompletionMode(QCompleter::PopupCompletion);
    m_completer->setCaseSensitivity(Qt::CaseInsensitive);
    m_completer->setFilterMode(Qt::MatchContains);
    
    // 设置completer的字体大小
    m_completer->popup()->setStyleSheet(QString("font-size: %1pt;").arg(ZenoStyle::dpiScaled(10)));

    m_completerModel = new QStringListModel(this);
    m_completer->setModel(m_completerModel);

    m_filterEdit->setCompleter(m_completer);
    
    // 清除列表按钮
    m_clearListButton = new QPushButton(tr("reset"));
    m_clearListButton->setProperty("cssClass", "shadowButton");
    m_filterSelectedNodesButton = new QPushButton(tr("filter selected nodes"));
    m_filterSelectedNodesButton->setProperty("cssClass", "shadowButton");
    QHBoxLayout* clearListButtonLayout = new QHBoxLayout();
    clearListButtonLayout->addWidget(m_filterSelectedNodesButton);
    clearListButtonLayout->addStretch();
    clearListButtonLayout->addWidget(m_clearListButton);

    // 节点列表
    m_nodeListView = new QListView;
    m_nodeListView->setModel(m_nodeListModel);
    m_nodeListView->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_nodeListView->setStyleSheet(QString("font-size: %1pt;").arg(ZenoStyle::dpiScaled(10)));

    m_filterEdit->setMinimumWidth(1);
    m_clearListButton->setMinimumWidth(1);
    m_filterSelectedNodesButton->setMinimumWidth(1);
    
    leftLayout->addWidget(m_filterEdit);
    leftLayout->addWidget(m_nodeListView);
    leftLayout->addLayout(clearListButtonLayout);  // 添加清除按钮到布局
    
    // 右边部分
    QWidget *rightWidget = new QWidget;
    QVBoxLayout *rightLayout = new QVBoxLayout(rightWidget);
    rightLayout->setContentsMargins(1,0,0,0);  // 减少内边距
    
    // 表格视图
    m_benchmarkTableView = new QTableView;
    m_benchmarkTableView->setModel(m_tableProxyModel);
    m_benchmarkTableView->setSortingEnabled(true);
    m_benchmarkTableView->setSelectionBehavior(QAbstractItemView::SelectRows);
    m_benchmarkTableView->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
    m_benchmarkTableView->horizontalHeader()->setSectionResizeMode(5, QHeaderView::Stretch);

    rightLayout->addWidget(m_benchmarkTableView);
    
    // 将左右部件添加到分割器
    splitter->addWidget(leftWidget);
    splitter->addWidget(rightWidget);
    
    // 设置初始大小比例 (1:2)
    QList<int> sizes({200, 600});
    splitter->setSizes(sizes);
    splitter->setChildrenCollapsible(false);
    
    // 将分割器添加到主布局
    mainLayout->addLayout(topLayout);
    mainLayout->addWidget(splitter);
    
    setLayout(mainLayout);

    connect(m_filterCb, &QCheckBox::stateChanged, this, [leftWidget](int state) {
        leftWidget->setVisible(state == Qt::Checked);
        });
}

void zenoBenchmark::setupConnections() {
    connect(m_filterEdit, &QLineEdit::textChanged, this, &zenoBenchmark::onFilterTextChanged);
    connect(m_completer, QOverload<const QString &>::of(&QCompleter::activated),  this, &zenoBenchmark::onCompleterActivated);
    connect(m_nodeListView, &QListView::doubleClicked, this, &zenoBenchmark::onListViewDoubleClicked);
    connect(m_clearListButton, &QPushButton::clicked, this, [this]() {
        m_nodeListModel->clear();
        });  // 连接清除按钮点击信号
    connect(m_filterSelectedNodesButton, &QPushButton::clicked, this, [this]() {
        auto main = zenoApp->getMainWindow();
        ZASSERT_EXIT(main);
        auto editor = main->getAnyEditor();
        ZASSERT_EXIT(editor);
        auto view = editor->getCurrentSubGraphView();
        ZASSERT_EXIT(view);
        auto scene = view->scene();
        ZASSERT_EXIT(scene);
        QList<QString> resStrList;
        for (auto item : scene->selectedItems()) {
            if (ZenoNode* pNode = qgraphicsitem_cast<ZenoNode*>(item)) {
                m_nodeListModel->addNode(pNode->index().data(ROLE_OBJID).toString());
            }
        }
        });
    connect(m_searchEdit, &QLineEdit::textChanged, this, [this](const QString &text) {
        m_tableProxyModel->setFilterKeyColumn(5);
        m_tableProxyModel->setFilterCaseSensitivity(Qt::CaseInsensitive);
        if (text.isEmpty()) {
            m_tableProxyModel->setFilterFixedString("");
        } else {
            m_tableProxyModel->setFilterFixedString(text);
        }
    });
}

void zenoBenchmark::onFilterTextChanged(const QString &text) {
    if (m_completer->popup()->isVisible()) {
        return;
    }
    
    if (IGraphsModel* pModel = GraphsManagment::instance().currentModel()) {
        QList<QString> resStrList;
        for (SEARCH_RESULT res : pModel->search(text, SEARCH_NODEID, SEARCH_FUZZ)) {
            resStrList.append(res.targetIdx.data(ROLE_OBJID).toString());
        }
        m_completerModel->setStringList(resStrList);
    }
}

void zenoBenchmark::onCompleterActivated(const QString &text) {
    m_nodeListModel->addNode(text);
     m_filterEdit->setCompleter(nullptr);
     m_filterEdit->clear();
     m_filterEdit->setCompleter(m_completer);
}

void zenoBenchmark::onListViewDoubleClicked(const QModelIndex &index) {
    m_nodeListModel->removeNode(index.row());
}