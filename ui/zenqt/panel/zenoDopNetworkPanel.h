#pragma once

#include <QtWidgets>

class zenoDopNetworkPanel  : public QTabWidget
{
    Q_OBJECT

public:
    zenoDopNetworkPanel(QWidget* inputsWidget, QWidget *parent);
    ~zenoDopNetworkPanel();
};
