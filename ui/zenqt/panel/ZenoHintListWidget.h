#pragma once

#include <QtWidgets>

#define SideLength 12

class TriangleButton : public QWidget {
    Q_OBJECT
public:
    TriangleButton(const QString& text, QWidget* parent = nullptr) : QWidget(parent) {
        setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
        path.moveTo(SideLength, 0);
        path.lineTo(SideLength, SideLength);
        path.lineTo(0, SideLength);
        path.lineTo(SideLength, 0);
    }

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter painter(this);
        painter.fillPath(path, Qt::black);
    }
private:
    QPainterPath path;
};

class ZenoHintListWidget : public QWidget {
    Q_OBJECT
public:
    ZenoHintListWidget();
    void setData(QStringList items);
    void setActive();
    void clearCurrentItem();

public slots:
    void sltItemSelect(const QModelIndex& selectedIdx);
signals:
    void hintSelected(QString);
    void escPressedHide();
    void clickOutSideHide();

protected:
    bool eventFilter(QObject* watched, QEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event)override;
    void paintEvent(QPaintEvent* event) override;
private:
    bool m_resizing = false;

    QListView* m_listView;
    QStringListModel* m_model;
    QWidget* m_button;
};