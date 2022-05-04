#ifndef __INNER_SHADOW_EFFECT_H__
#define __INNER_SHADOW_EFFECT_H__

#include <QGraphicsEffect>

class InnerShadowEffect : public QGraphicsEffect
{
    Q_OBJECT
public:
    InnerShadowEffect(QObject* parent = nullptr);
    ~InnerShadowEffect();
    QRectF boundingRectFor(const QRectF& sourceRect) const override;

protected:
    void draw(QPainter *painter) override;
    void sourceChanged(ChangeFlags flags) override;
};


#endif