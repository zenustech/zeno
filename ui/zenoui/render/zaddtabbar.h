#ifndef __ZADD_TABBAR_H__
#define __ZADD_TABBAR_H__

#include <QtWidgets>

class ZIconToolButton;
class ZIconLabel;

class ZAddTabBar : public QTabBar
{
    Q_OBJECT
public:
    explicit ZAddTabBar(QWidget* parent = nullptr);
    ~ZAddTabBar();
    QSize sizeHint() const override;

signals:
    void addBtnClicked();
    void layoutBtnClicked();

protected:
    void resizeEvent(QResizeEvent* event) override;
    void tabLayoutChange() override;
    void mousePressEvent(QMouseEvent* e) override;
    void mouseReleaseEvent(QMouseEvent* e) override;
    void paintEvent(QPaintEvent* e) override;

private:
    void setGeomForAddBtn();
    void setGeomForLayoutBtn();

    //ZIconToolButton* m_pAddBtn;
    ZIconLabel* m_pAddBtn;
    ZIconLabel* m_pLayoutBtn;
};


#endif