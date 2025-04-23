#pragma once

#include <QtWidgets>

QT_BEGIN_NAMESPACE
namespace Ui { class ZComposeVideoDlgClass; };
QT_END_NAMESPACE

class ZComposeVideoDlg : public QDialog
{
    Q_OBJECT

public:
    ZComposeVideoDlg(QWidget *parent = nullptr);
    ~ZComposeVideoDlg();

    bool combineVideo();

public slots:
    void onAcceptClicked();

private:
    Ui::ZComposeVideoDlgClass * m_ui;
};
