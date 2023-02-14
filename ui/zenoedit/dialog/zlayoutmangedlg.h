#pragma once

#include <QDialog>
#include "ui_zlayoutMangedlg.h"

QT_BEGIN_NAMESPACE
namespace Ui { class ZLayoutMangeDlgClass; };
QT_END_NAMESPACE

class ZLayoutMangeDlg : public QDialog
{
    Q_OBJECT

public:
    ZLayoutMangeDlg(QWidget *parent = nullptr);
    ~ZLayoutMangeDlg();
signals:
    void layoutChangedSignal();
  private:
    void initUI();

  private:
    Ui::ZLayoutMangeDlgClass *ui;
};
