#ifndef __ZRECORDDLG_H__
#define __ZRECORDDLG_H__

namespace Ui
{
	class RecordVideoDlg;
}

#include <QDialog>
#include "viewport/viewportwidget.h"
#include "viewport/recordvideomgr.h"

class ZPathEdit;
class ZRecordVideoDlg : public QDialog
{
	Q_OBJECT
public:
	ZRecordVideoDlg(QWidget* parent = nullptr);
	bool getInfo(VideoRecInfo &info);
private:
	Ui::RecordVideoDlg* m_ui;
    ZPathEdit* m_exePath;
};
#endif