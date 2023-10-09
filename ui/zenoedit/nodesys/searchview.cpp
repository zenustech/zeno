#include "searchview.h"

#include <QVBoxLayout>
#include "zenoui/style/zenostyle.h"


SearchResultItem::SearchResultItem(QWidget* parent)
    : QLabel(parent)
{
    this->setProperty("cssClass", "search_result");
}

void SearchResultItem::setResult(const QString& result, const QVector<int>& matchIndices, const QString& category, bool enableCategory)
{
    bool deprecated = category == "deprecated";

    auto getHighlightHtml = [](const QString& text) {
        return QString("<b style='color:DeepSkyBlue;'>%1</b>").arg(text);
    };

    auto getNormalHtml = [deprecated](const QString& text) {
        if (!deprecated) {
            return text;
        }
        return QString("<span style='color:DarkGray;'>%1</span>").arg(text);
    };

    QString htmlText;
    int left = 0;
    for (int index : matchIndices) {
        // [left, index - 1] is normal
        auto normalText = result.mid(left, index - left);
        if (!normalText.isEmpty()) {
            htmlText += getNormalHtml(normalText);
        }
        // [index, index] is highlight
        auto highLightText = result.mid(index, 1);
        htmlText += getHighlightHtml(highLightText);

        left = index + 1;
    }

    auto normalText = result.mid(left, result.size() - left);
    if (!normalText.isEmpty()) {
        htmlText += getNormalHtml(normalText);
    }

    if (enableCategory) {
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
    setUniformItemSizes(true);
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
    itemWidget->setResult(text, matchIndices, category, m_enableCategory);
}

void SearchResultWidget::resizeCount(int count)
{
    while (this->count() < count) {
        int index = this->count();
        this->addItem("");
        auto itemWidget = new SearchResultItem(this);
        auto modelIndex = this->model()->index(this->count() - 1, 0);
        this->setIndexWidget(modelIndex, itemWidget);
    }
    while (this->count() > count) {
        auto item = this->item(this->count() - 1);
        this->removeItemWidget(item);
        delete item; // delete will remove from list
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
    auto width = std::max(sizeHintForColumn(0), (int)ZenoStyle::dpiScaled(300));
    return QSize(width, height);
}
