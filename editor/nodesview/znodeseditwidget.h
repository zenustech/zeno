#ifndef __ZNODES_EDITWIDGET_H__
#define __ZNODES_EDITWIDGET_H__

#include <QtWidgets>

class ZenoGraphsWidget;

class ZNodesEditWidget : public QWidget
{
    Q_OBJECT
public:
    ZNodesEditWidget(QWidget* parent = nullptr);

public slots:
    void openFileDialog();

private slots:
    void onSubGraphTriggered();

private:
    void initMenu(QMenuBar* pMenu);

    ZenoGraphsWidget* m_pGraphsWidget;
    QComboBox* m_pComboSubGraph;
    QPushButton* m_pNewBtn;
    QPushButton* m_pDeleteBtn;
    QAction* m_pNewSubGraph;
};


#endif