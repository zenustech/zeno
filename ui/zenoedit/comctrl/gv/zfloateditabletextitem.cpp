#include "zfloateditabletextitem.h"
#include "zassert.h"
#include <zenomodel/include/modeldata.h>
#include "util/apphelper.h"

//float
ZFloatEditableTextItem::ZFloatEditableTextItem(const QString &text, QGraphicsItem *parent):
    _base(text, parent)
{
}
ZFloatEditableTextItem::ZFloatEditableTextItem(QGraphicsItem *parent) :
    _base(parent)
{
}

bool ZFloatEditableTextItem::event(QEvent *event) 
{
    ZenoMainWindow *mainWin = zenoApp->getMainWindow();
    ZASSERT_EXIT(mainWin, false);
    ZTimeline *timeline = mainWin->timeline();
    ZASSERT_EXIT(timeline, false);
    CURVE_DATA curve;
    {
        if (event->type() == QEvent::DynamicPropertyChange) 
        {
            QDynamicPropertyChangeEvent *evt = static_cast<QDynamicPropertyChangeEvent *>(event);
            if (evt->propertyName() == g_keyFrame) 
            {
                updateText(timeline->value());
                if (AppHelper::getKeyFrame(this, curve)) {
                    connect(timeline, &ZTimeline::sliderValueChanged, this, &ZFloatEditableTextItem::updateText, Qt::UniqueConnection);
                    connect(zenoApp->getMainWindow(), &ZenoMainWindow::visFrameUpdated, this, &ZFloatEditableTextItem::onUpdate, Qt::UniqueConnection);
                } else {
                    disconnect(timeline, &ZTimeline::sliderValueChanged, this, &ZFloatEditableTextItem::updateText);
                    disconnect(zenoApp->getMainWindow(), &ZenoMainWindow::visFrameUpdated, this, &ZFloatEditableTextItem::onUpdate);
                }
            }
        }
        if (event->type() == QEvent::FocusOut) {
            updateCurveData();
        }
    }
    return _base::event(event);
}
void ZFloatEditableTextItem::updateCurveData() {
    CURVE_DATA val;
    if (!AppHelper::getKeyFrame(this, val)) {
        return;
    }
    ZenoMainWindow *mainWin = zenoApp->getMainWindow();
    ZASSERT_EXIT(mainWin);
    ZTimeline *timeline = mainWin->timeline();
    ZASSERT_EXIT(timeline);
    float x = timeline->value();
    float y = text().toFloat();
    if (val.visible) {
        bool bUpdate = curve_util::updateCurve(QPoint(x, y), val);
        if (bUpdate)
            setProperty(g_keyFrame, QVariant::fromValue(val));
    } else {
        val.points.begin()->point = QPointF(x, y);
        setProperty(g_keyFrame, QVariant::fromValue(val));
    }
}

void ZFloatEditableTextItem::updateText(int frame) 
{
    CURVE_DATA data;
    if (AppHelper::getKeyFrame(this, data)) {
        QString text = QString::number(data.eval(frame));
        setText(text);
    }
    AppHelper::updateProperty(this);
    update();
}

void ZFloatEditableTextItem::onUpdate(bool gl, int frame) 
{
    updateText(frame);
}
