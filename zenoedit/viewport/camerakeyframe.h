#ifndef __CAMERA_KEYFRAME_H__
#define __CAMERA_KEYFRAME_H__

#include <QtWidgets>
#include "zenovis.h"

class CameraKeyframeWidget : public QWidget
{
public:
    CameraKeyframeWidget();
    bool queryFrame(int frame, PerspectiveInfo& ret);
    void insertKeyFrames();
    void removeKeyFrame();
    void updateList();

private:
    QListWidget* m_list;
    QComboBox* m_enable;
    QPushButton* m_key;
    QPushButton* m_remove;
    std::map<int, PerspectiveInfo> m_keyFrames;
};


#endif