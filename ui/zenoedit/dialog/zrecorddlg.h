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
	ZRecordVideoDlg(QWidget* parent = nullptr);
	bool getInfo(int& fps, int& bitrate, float& width, float& height,
                 QString& path, QString& fn, int &numOptix, int &numMSAA, bool& bExportVideo, bool& needDenoise);

private:
	Ui::RecordVideoDlg* m_ui;
};

#endif