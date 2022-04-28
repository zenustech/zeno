#ifndef __CURVE_MODEL_H__
#define __CURVE_MODEL_H__

#include <QStandardItemModel>
#include "../curvemap/curveutil.h"
#include <zenoui/model/modeldata.h>

enum CURVE_ROLE
{
    ROLE_NODEPOS = Qt::UserRole + 1,   //logic pos
    ROLE_LEFTPOS,   //left handle pos offset,
    ROLE_RIGHTPOS,  //right handle pos
    ROLE_TYPE,
    ROLE_LOCKX,
    ROLE_LOCKY
};

enum HANDLE_TYPE
{
    HDL_FREE,
    HDL_ALIGNED,
    HDL_VECTOR,
    HDL_ASYM
};

class CurveModel : public QStandardItemModel
{
    Q_OBJECT
public:
    explicit CurveModel(const QString& id, const CURVE_RANGE& rg, QObject *parent = nullptr);
    CurveModel(const QString& id, const CURVE_RANGE& rg, int rows, int columns, QObject *parent = nullptr);
    ~CurveModel();
    //method for temporary node like MakeCurvemap, DynamicNumber¡£
    void initItems(CURVE_RANGE rg, const QVector<QPointF>& points, const QVector<QPointF>& handlers);
    bool setData(const QModelIndex &index, const QVariant &value, int role = Qt::EditRole) override;
    void resetRange(const CURVE_RANGE& rg);
    CURVE_RANGE range() const;
    QString id() const;
    QPointF clipNodePos(const QModelIndex& idx, const QPointF& currPos);

    QPointF adjustLastRightHdl(const QModelIndex& currIdx);
    QPointF adjustCurrLeftHdl(const QModelIndex& currIdx);
    QPointF adjustNextLeftHdl(const QModelIndex& currIdx);
    QPointF adjustCurrRightHdl(const QModelIndex& currIdx);
    QPair<QPointF, QPointF> adjustWhenLeftHdlChanged(const QModelIndex& currIdx, QPointF newPos);
    QPair<QPointF, QPointF> adjustWhenRightHdlChanged(const QModelIndex& currIdx, QPointF newPos);

signals:
    void rangeChanged(CURVE_RANGE);

private:
    CURVE_RANGE m_range;
    QString m_id;
};

#endif