#ifndef __NODE_EDITOR_TABWIDGET_H__
#define __NODE_EDITOR_TABWIDGET_H__

#include <QtWidgets>

class NodeEditorTabWidget : public QTabWidget
{
    Q_OBJECT
public:
    explicit NodeEditorTabWidget(QWidget* parent = nullptr);
    ~NodeEditorTabWidget();

protected:
    void paintEvent(QPaintEvent* e) override;
    void enterEvent(QEvent* event) override;
    void leaveEvent(QEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    bool eventFilter(QObject* watched, QEvent* event) override;

signals:
    void addClicked();
    void layoutBtnClicked();
    void tabAboutToClose(int i);
    void tabClosed(int i);
};

#endif