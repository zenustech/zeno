#ifndef __ZNODES_EDITWIDGET_H__
#define __ZNODES_EDITWIDGET_H__

#include <QtWidgets>

class ZenoGraphsWidget;
class GraphsModel;

class ZNodesEditWidget : public QWidget
{
    Q_OBJECT
public:
    ZNodesEditWidget(QWidget* parent = nullptr);

public slots:
    void openFileDialog();
    void saveAs();

private slots:
    void onSubGraphTriggered();
    void importGraph();
    void exportGraph();

private:
    void initMenu(QMenuBar* pMenu);
    QString getOpenFileByDialog();

    ZenoGraphsWidget* m_pGraphsWidget;
    QComboBox* m_pComboSubGraph;
    QPushButton* m_pReloadBtn;
    QPushButton* m_pDeleteBtn;
    QAction* m_pNewSubGraph;

    GraphsModel* m_model;
};


#endif