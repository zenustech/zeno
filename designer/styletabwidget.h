#ifndef __STYLE_TABWIDGET_H__
#define __STYLE_TABWIDGET_H__

class StyleTabWidget : public QTabWidget
{
    Q_OBJECT
public:
    StyleTabWidget(QWidget* parent = nullptr);

signals:
    void tabClosed(int);

private slots:
    void onTabClosed(int);
};

#endif