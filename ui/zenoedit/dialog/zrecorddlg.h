#ifndef __ZRECORDDLG_H__
#define __ZRECORDDLG_H__

namespace Ui
{
	class RecordVideoDlg;
}

#include <QDialog>
#include "viewport/viewportwidget.h"

class ZRecordVideoDlg : public QDialog
{
	Q_OBJECT
public:
	ZRecordVideoDlg(int frameStart, int frameEnd, QWidget* parent = nullptr);
	bool getInfo(int& frameStart, int& frameEnd, int& fps, int& bitrate, float& width, float& height,
                 QString& path, QString& fn, int &numOptix, int &numMSAA,
                 bool& bRecordWhenRun, bool& bExportVideo);

private:
	Ui::RecordVideoDlg* m_ui;
};

#endif