#pragma once

#include <QtWidgets>
#include "widgets/zlineedit.h"
#include <zeno/core/data.h>

#define SideLength 12
#define minWidth 80
#define minHeight 150

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

    void resetCurrentItem();
    void clearCurrentItem();

    void resetSize();
    void setCurrentZlineEdit(ZLineEdit* linedit);

    QString getCurrentText();

    void setCalcPropPanelPosFunc(std::function<QPoint()> func);
    QPoint getPropPanelPos();

public slots:
    void sltItemSelect(const QModelIndex& selectedIdx);
signals:
    void hintSelected(QString);
    void resizeFinished();

    void escPressedHide();
    void clickOutSideHide(QWidget*);

protected:
    bool eventFilter(QObject* watched, QEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event)override;
    void mouseReleaseEvent(QMouseEvent* event)override;
    void paintEvent(QPaintEvent* event) override;
private:
    bool m_resizing = false;

    QListView* m_listView;
    QStringListModel* m_model;
    QWidget* m_button;

    ZLineEdit* m_currentLineEdit;

    std::function<QPoint()> m_getPropPanelPosfunc;
};

class ZenoFuncDescriptionLabel :public QWidget
{
    Q_OBJECT
public:
    ZenoFuncDescriptionLabel();
    void setDesc(zeno::FUNC_INFO func, int pos);
    void setCurrentFuncName(std::string funcName);
    std::string getCurrentFuncName();
    void setCalcPropPanelPosFunc(std::function<QPoint()> func);
    QPoint getPropPanelPos();

protected:
    void paintEvent(QPaintEvent* event) override;
    bool eventFilter(QObject* watched, QEvent* event) override;

private:
    QLabel* m_label;
    std::string m_currentFunc;

    std::function<QPoint()> m_getPropPanelPosfunc;
};