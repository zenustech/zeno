#ifndef __CURVE_MODEL_H__
#define __CURVE_MODEL_H__

#include <QStandardItemModel>
#include "uicommon.h"

enum CURVE_ROLE
{
    ROLE_NODEPOS = Qt::UserRole + 1,   //logic pos
    ROLE_LEFTPOS,   //left handle pos offset,
    ROLE_RIGHTPOS,  //right handle pos
    ROLE_TYPE,
    ROLE_LOCKX,
    ROLE_LOCKY,
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
    explicit CurveModel(const QString& id, const zeno::CurveData::Range& rg, QObject *parent = nullptr);
    CurveModel(const QString& id, const zeno::CurveData::Range& rg, int rows, int columns, QObject *parent = nullptr);
    ~CurveModel();
    //method for temporary node like MakeCurvemap, DynamicNumber
    void initItems(zeno::CurveData const &curvedat);
    zeno::CurveData getItems() const;
    bool setData(const QModelIndex &index, const QVariant &value, int role = Qt::EditRole) override;
    void resetRange(const zeno::CurveData::Range& rg);
    zeno::CurveData::Range range() const;
    bool isTimeline() const;
    void setTimeline(bool bTimeline);
    QString id() const;
    void setId(QString id);
    void setVisible(bool visible);
    bool getVisible();
    std::string z_serialize() const;
    void z_deserialize(std::string_view s);
    QPointF clipNodePos(const QModelIndex& idx, const QPointF& currPos);

    QPointF adjustLastRightHdl(const QModelIndex& currIdx);
    QPointF adjustCurrLeftHdl(const QModelIndex& currIdx);
    QPointF adjustNextLeftHdl(const QModelIndex& currIdx);
    QPointF adjustCurrRightHdl(const QModelIndex& currIdx);
    QPair<QPointF, QPointF> adjustWhenLeftHdlChanged(const QModelIndex& currIdx, QPointF newPos);
    QPair<QPointF, QPointF> adjustWhenRightHdlChanged(const QModelIndex& currIdx, QPointF newPos);

signals:
    void rangeChanged(zeno::CurveData::Range);

private:
    zeno::CurveData::Range m_range;
    QString m_id;
    bool m_bTimeline;
    bool m_bVisible;
};

typedef QMap<QString, CurveModel*> CURVES_MODEL;
Q_DECLARE_METATYPE(CURVES_MODEL)

#endif
