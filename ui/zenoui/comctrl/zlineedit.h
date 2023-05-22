#ifndef __ZLINEEDIT_H__
#define __ZLINEEDIT_H__

#include <QtWidgets>
#include <zenomodel/include/modeldata.h>

class ZNumSlider;
class ZTimeline;

class ZLineEdit : public QLineEdit
{
    Q_OBJECT
public:
    explicit ZLineEdit(QWidget* parent = nullptr);
    explicit ZLineEdit(const QString& text, QWidget* parent = nullptr);
    void setNumSlider(const QVector<qreal>& steps);
    void setShowingSlider(bool bShow);
    void setIcons(const QString& icNormal, const QString& icHover);

protected:
    bool event(QEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void keyPressEvent(QKeyEvent* event) override;
    void keyReleaseEvent(QKeyEvent* event) override;
    void paintEvent(QPaintEvent* event) override;
    bool eventFilter(QObject *obj, QEvent *event) override;

signals:
    void btnClicked();
    void textEditFinished();

private:
    void popupSlider();
    void init();

    QVector<qreal> m_steps;
    ZNumSlider* m_pSlider;
    bool m_bShowingSlider;
    bool m_bHasRightBtn;
    QPushButton *m_pButton;
    QString m_iconNormal;
    QString m_iconHover;
    bool m_bIconHover;
};

struct CURVE_DATA;
class ZFloatLineEdit : public ZLineEdit {
    Q_OBJECT
  public:
    explicit ZFloatLineEdit(QWidget *parent = nullptr);
    explicit ZFloatLineEdit(const QString &text, QWidget *parent = nullptr);
    void getDelfCurveData(CURVE_DATA &val, bool bVisible = true);
    void updateCurveData();
    bool getKeyFrame(CURVE_DATA &curve);

  protected:
    bool event(QEvent *event) override;
    void contextMenuEvent(QContextMenuEvent *) override;

  private slots:
    void updateBackgroundProp(int frame);

  private:
    ZTimeline *getTimeline();
    bool isSetKeyFrame();
    void updateHandler(CURVE_DATA &curve);
};
extern const char *g_keyFrame;
#endif
