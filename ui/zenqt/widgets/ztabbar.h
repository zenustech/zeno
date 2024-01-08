#ifndef __ZTABBAR_H__
#define __ZTABBAR_H__

#include <QtWidgets>

class ZIconButton;

struct Tab {
    QIcon icon;
    QString text;
    QRect rect;
    ZIconButton* pCloseButton;

    Tab(const QIcon& ico, const QString& txt) : icon(ico), text(txt), pCloseButton(nullptr) {}
};

class ZTabBar : public QWidget
{
    Q_OBJECT
public:
    explicit ZTabBar(QWidget* parent = nullptr);
    ~ZTabBar();

    Tab at(int index);
    int addTab(const QString& text);
    int addTab(const QIcon& icon, const QString& text);
    int insertTab(int index, const QString& text);
    int insertTab(int index, const QIcon& icon, const QString& text);
    void moveTab(int srcIndex, int dstIndex);
    void removeTab(int index);

    QString tabText(int index) const;
    void setTabText(int index, const QString& text);

public slots:
    void setCurrentIndex(int index);

signals:
    void currentChanged(int index);
    void tabCloseBtnClicked(int index);
    void tabMoved(int from, int to);
    void tabBarClicked(int index);

protected:
    void paintEvent(QPaintEvent*) override;

private:
    inline bool validIndex(int index) const { return index >= 0 && index < m_tabList.count(); }
    void refresh();
    void _closeTab();

    QList<Tab> m_tabList;
    int m_currentIndex;
    bool m_bCloseBtn;
};



#endif