#ifndef __ZENO_COMMON_H__
#define __ZENO_COMMON_H__

struct TIMELINE_INFO
{
    int beginFrame;
    int endFrame;
    int currFrame;
    bool bAlways;

    TIMELINE_INFO() : beginFrame(0), endFrame(0), currFrame(0), bAlways(false) {}
};

struct APP_SETTINGS
{
    TIMELINE_INFO timeline;
    //todo: other settings.
};

#endif