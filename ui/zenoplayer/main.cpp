#include <QApplication>
#include "zenoplayer.h"

#include "zenoapplication.h"
#include "style/zenostyle.h"

int main(int argc, char *argv[]) 
{
    ZenoApplication a(argc, argv);
    a.setStyle(new ZenoStyle);

	ZenoPlayer w;
	w.show();
	return a.exec();
}
