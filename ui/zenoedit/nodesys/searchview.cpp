#include "searchview.h"

#include <QVBoxLayout>
#include "zenoui/style/zenostyle.h"


SearchResultItem::SearchResultItem(QWidget* parent)
    : QLabel(parent)
{
    this->setProperty("cssClass", "search_result");
}

void SearchResultItem::setResult(const QString& result, const QVector<int>& matchIndices, const QString& category)
{
    QString htmlText;

    // matchIndices is sorted, so here can use index++
    int index = 0;
    for (int i = 0; i < result.size(); ++i) {
        if (index >= matchIndices.size() || i != matchIndices[index]) {
            htmlText += result[i];
        }
        else {
            htmlText += "<b style='color:DeepSkyBlue;'>";
            htmlText += result[i];
            htmlText += "</b>";
            index++;
        }
    }

    if (!category.isEmpty()) {
        htmlText += QString("<span style='color:DarkGray;font-size:15px;'> (%1)</span>").arg(category);
    }

    this->setText(htmlText);
    m_result = result;
}

QString SearchResultItem::result()
{
    return m_result;
}

SearchResultWidget::SearchResultWidget(QWidget* parent)
    : QListWidget(parent)
    , m_enableCategory(true)
{
    setResizeMode(QListView::Adjust);
    setProperty("cssClass", "search_view");

    // mouse click will emit pressed
    connect(this, &QListWidget::pressed, this, [this](const QModelIndex& index) {
         auto itemWidget = qobject_cast<SearchResultItem*>(this->indexWidget(index));
         emit clicked(itemWidget);
    });
    // press return will emit activated
    connect(this, &QListWidget::activated, this, [this](const QModelIndex& index) {
        auto itemWidget = qobject_cast<SearchResultItem*>(this->indexWidget(index));
        emit clicked(itemWidget);
    });
}

void SearchResultWidget::setEnableCategory(bool enable)
{
    m_enableCategory = enable;
}

void SearchResultWidget::setResult(int row, const QString& text, const QVector<int>& matchIndices, const QString& category)
{
    if (row < 0 || row >= this->count()) return;

    auto modelIndex = this->model()->index(row, 0);
    auto itemWidget = qobject_cast<SearchResultItem*>(this->indexWidget(modelIndex));
    itemWidget->setResult(text, matchIndices, m_enableCategory ? category : "");
}

void SearchResultWidget::resizeCount(int count)
{
    while (this->count() < count) {
        this->addItem("");
        auto itemWidget = new SearchResultItem(this);
        auto modelIndex = this->model()->index(this->count() - 1, 0);
        this->setIndexWidget(modelIndex, itemWidget);
    }
    while (this->count() > count) {
        auto item = this->takeItem(this->count() - 1);
        if (item) {
            delete item;
        }
    }
}

void SearchResultWidget::moveToTop()
{
    setCurrentIndex(model()->index(0, 0));
    clearSelection();
}

void SearchResultWidget::keyPressEvent(QKeyEvent* event)
{
    QListWidget::keyPressEvent(event);
}

void SearchResultWidget::resizeEvent(QResizeEvent* event)
{
    QListWidget::resizeEvent(event);
}

QSize SearchResultWidget::sizeHint() const
{ 
    int rowCount = count();
    if (rowCount == 0) return QSize(0, 0);

    auto height = std::min(10, rowCount) * sizeHintForRow(0) + 10;
    auto width = std::max(sizeHintForColumn(0), 450);
    return QSize(width, height);
}
