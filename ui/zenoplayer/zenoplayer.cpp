#include "zenoplayer.h"
#include "../zenoedit/viewport/viewportwidget.h"

ZenoPlayer::ZenoPlayer(QWidget* parent)
    : QWidget(parent)
{

   DisplayWidget *view = new DisplayWidget;

   QHBoxLayout* layMain = new QHBoxLayout;
   layMain->setMargin(0);
   layMain->setSpacing(0);
   layMain->addWidget(view);
   setLayout(layMain);

}

ZenoPlayer::~ZenoPlayer()
{

}
