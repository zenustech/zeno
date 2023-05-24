#pragma once

#include <QDialog>

QT_BEGIN_NAMESPACE
namespace Ui { class ZLayoutMangeDlgClass; };
QT_END_NAMESPACE

class ZDockLayoutMangeDlg : public QDialog
{
    Q_OBJECT

public:
    ZDockLayoutMangeDlg(QWidget *parent = nullptr);
    ~ZDockLayoutMangeDlg();
signals:
    void layoutChangedSignal();
  private:
    void initUI();

  private:
    Ui::ZLayoutMangeDlgClass *ui;
};
