#pragma once

#include <QtWidgets>
#include <QAbstractItemModel>
#include <QSortFilterProxyModel>
#include <unordered_set>

// 列表模型
class NodeListModel : public QAbstractListModel
{
    Q_OBJECT

public:
    explicit NodeListModel(QObject *parent = nullptr);
    int rowCount(const QModelIndex &parent = QModelIndex()) const override;
    QVariant data(const QModelIndex &index, int role = Qt::DisplayRole) const override;
    void addNode(const QString &node);
    void removeNode(int index);
    void clear();
    QStringList getNodes() const;

private:
    QStringList m_nodes;
};

// 表格模型
class BenchmarkTableModel : public QAbstractTableModel
{
    Q_OBJECT

public:
    explicit BenchmarkTableModel(QObject *parent = nullptr);
    int rowCount(const QModelIndex &parent = QModelIndex()) const override;
    int columnCount(const QModelIndex &parent = QModelIndex()) const override;
    QVariant data(const QModelIndex &index, int role = Qt::DisplayRole) const override;
    QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const override;
    void sort(int column, Qt::SortOrder order = Qt::AscendingOrder) override;
    void clear();
    void reset(const std::string& data);

private:
    struct BenchmarkData {
        int avg;
        int min;
        int max;
        int total;
        int cnt;
        QString tag;
    };
    QVector<BenchmarkData> m_data;
};

class zenoBenchmark : public QWidget
{
    Q_OBJECT

public:
    zenoBenchmark(QWidget *parent = nullptr);
    ~zenoBenchmark();

    std::string monitoredNodes();
    void setBenchMarkData(std::string data);

private slots:
    void onFilterTextChanged(const QString &text);
    void onCompleterActivated(const QString &text);
    void onListViewDoubleClicked(const QModelIndex &index);

private:
    void setupUI();
    void setupConnections();

    QCheckBox* m_filterCb;
    QLineEdit* m_filterEdit;
    QLabel* m_serachLabel;
    QLineEdit* m_searchEdit;
    QListView *m_nodeListView;
    QTableView *m_benchmarkTableView;
    QCompleter *m_completer;
    QPushButton *m_clearListButton,* m_filterSelectedNodesButton;  // 添加清除按钮成员变量
    
    NodeListModel *m_nodeListModel;
    BenchmarkTableModel *m_benchmarkTableModel;
    QSortFilterProxyModel *m_tableProxyModel;
    QStringListModel* m_completerModel;
};
