#pragma once

#include <QListWidget>
#include <QLabel>

class SearchResultItem : public QLabel
{
    Q_OBJECT
public:
    explicit SearchResultItem(QWidget* parent = nullptr);

    void setResult(const QString &result, const QVector<int> &matchIndices,
                   const QString &category, bool enableCategory);
    QString result();

private:
    QString m_result;
};

class SearchResultWidget : public QListWidget
{
    Q_OBJECT
public:
    explicit SearchResultWidget(QWidget* parent = nullptr);

    void setEnableCategory(bool enable);

    void setResult(int row, const QString& text, const QVector<int>& matchIndices, const QString& category);

    void resizeCount(int count);

    void moveToTop();

    QSize sizeHint() const override;

signals:
    void clicked(SearchResultItem* item);

protected:
    void keyPressEvent(QKeyEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;

private:
    QVector<QString> m_listData;
    bool m_enableCategory;
};

