#include "zfloatlineedit.h"
#include <zenoui/comctrl/dialog/curvemap/zcurvemapeditor.h>
#include "util/apphelper.h"

//FLOAT LINEEDIT
ZFloatLineEdit::ZFloatLineEdit(QWidget *parent):
    ZLineEdit(parent)
{
    setProperty(g_setKey, "null");
}
ZFloatLineEdit::ZFloatLineEdit(const QString &text, QWidget *parent) : 
    ZLineEdit(text, parent)
{
    setProperty(g_setKey, "null");
}

void ZFloatLineEdit::updateCurveData() 
{
    CURVE_DATA val;
    if (!AppHelper::getKeyFrame(this, val)) {
        return;
    }
    if (ZTimeline *timeline = getTimeline()) {
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
}

bool ZFloatLineEdit::event(QEvent *event) 
{
    CURVE_DATA curve;
    {
        ZTimeline *timeline = getTimeline();
        ZASSERT_EXIT(timeline, false);
        if (event->type() == QEvent::DynamicPropertyChange) {
            QDynamicPropertyChangeEvent *evt = static_cast<QDynamicPropertyChangeEvent*>(event); 
            if (evt->propertyName() == g_keyFrame) {
                updateBackgroundProp(timeline->value());
                if (AppHelper::getKeyFrame(this, curve)) {
                    connect(timeline, &ZTimeline::sliderValueChanged, this, &ZFloatLineEdit::updateBackgroundProp, Qt::UniqueConnection);
                    connect( zenoApp->getMainWindow(), &ZenoMainWindow::visFrameUpdated, this, &ZFloatLineEdit::onUpdate, Qt::UniqueConnection);
                } else {
                    disconnect(timeline, &ZTimeline::sliderValueChanged, this, &ZFloatLineEdit::updateBackgroundProp);
                    disconnect( zenoApp->getMainWindow(), &ZenoMainWindow::visFrameUpdated, this, &ZFloatLineEdit::onUpdate);
                }
            }
        } 
        else if (event->type() == QEvent::FocusOut) {
            updateCurveData();
        }
    }
    ZLineEdit::event(event);
}

void ZFloatLineEdit::updateBackgroundProp(int frame) 
{
    CURVE_DATA data;
    if (AppHelper::getKeyFrame(this, data)) {
        QString text = QString::number(data.eval(frame));
        setText(text);
        
    }
    AppHelper::updateProperty(this);
    this->style()->unpolish(this);
    this->style()->polish(this);
    update();
}
void ZFloatLineEdit::onUpdate(bool gl, int frame) 
{
    updateBackgroundProp(frame);
}
ZTimeline *ZFloatLineEdit::getTimeline() 
{
    ZenoMainWindow *mainWin = zenoApp->getMainWindow();
    ZASSERT_EXIT(mainWin, nullptr);
    ZTimeline *timeline = mainWin->timeline();
    ZASSERT_EXIT(timeline, nullptr);
    return timeline;
}
