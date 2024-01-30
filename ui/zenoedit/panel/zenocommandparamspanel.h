#ifndef __ZENO_COMMANDPARAMSPANEL_H__
#define __ZENO_COMMANDPARAMSPANEL_H__

#include <QtWidgets>
#include <zenomodel/include/modeldata.h>

class ZenoCommandParamsPanel : public QWidget
{
    Q_OBJECT
public:
    ZenoCommandParamsPanel(QWidget *parent = nullptr);
protected:
    void keyPressEvent(QKeyEvent* e) override;
private slots:
    void onExport();
    void onItemClicked(QTableWidgetItem* item);
    void onItemChanged(QTableWidgetItem* item);
    void onUpdateCommandParams(const QString& path);
    void onModelClear();
    void onModelInited();
private:
    void initUi();
    void initConnection();
    void appendRow(const QStringList&path, const CommandParam& val);
    void initTableWidget();
private:
    QTableWidget* m_pTableWidget;
    QPushButton* m_pExportButton;
};

#endif

