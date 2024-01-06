#ifndef __ZENO_OPENPATHPANEL_H__
#define __ZENO_OPENPATHPANEL_H__

#include <QtWidgets>

class ZenoOpenPathPanel : public QWidget
{
    Q_OBJECT
public:
    ZenoOpenPathPanel(QWidget *parent = nullptr);
protected:
    bool eventFilter(QObject* obj, QEvent* evt) override;
private:
    void initUI();
private:
    QFileDialog* m_pFileDialog;
    QVBoxLayout* m_pLayout;
};

#endif

