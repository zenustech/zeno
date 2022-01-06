#ifndef __VIEW_MDIAREA_H__
#define __VIEW_MDIAREA_H__

class ViewMdiArea : public QMdiArea
{
    Q_OBJECT
public:
    ViewMdiArea(QMdiArea* parent = nullptr);
};

#endif