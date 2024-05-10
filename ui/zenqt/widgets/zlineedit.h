#ifndef __ZLINEEDIT_H__
#define __ZLINEEDIT_H__

#include <QtWidgets>

class ZTimeline;

class ZLineEdit : public QLineEdit
{
    Q_OBJECT
public:
    explicit ZLineEdit(QWidget* parent = nullptr);
    explicit ZLineEdit(const QString& text, QWidget* parent = nullptr);
    void setIcons(const QString& icNormal, const QString& icHover);

protected:
    void mouseReleaseEvent(QMouseEvent* event) override;
    void keyPressEvent(QKeyEvent* event) override;
    void paintEvent(QPaintEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;
    bool eventFilter(QObject *obj, QEvent *event) override;

    void resizeEvent(QResizeEvent* event) override;

signals:
    void btnClicked();
    void textEditFinished();

public slots:
    void sltHintSelected(QString itemSelected);
    void sltSetFocus();

private:
    void init();

    bool m_bHasRightBtn;
    QPushButton *m_pButton;
    QString m_iconNormal;
    QString m_iconHover;
    bool m_bIconHover;

    bool m_bShowHintList;
};

#endif
