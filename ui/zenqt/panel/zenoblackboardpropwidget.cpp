#include "zenoblackboardpropwidget.h"
#include "uicommon.h"
#include "widgets/zlabel.h"
#include "style/zenostyle.h"
#include <QGridLayout>
#include "zenoapplication.h"
#include "widgets/zwidgetfactory.h"
#include "model/graphsmanager.h"
#include "zenomainwindow.h"
#include "util/uihelper.h"
#include "variantptr.h"

ZenoBlackboardPropWidget::ZenoBlackboardPropWidget(const QPersistentModelIndex &index, QWidget *parent)
    : QWidget(parent), 
    m_idx(index), 
    m_pColor(nullptr),  
    m_pTitle(nullptr)
{
    QGridLayout *pGroupLayout = new QGridLayout(this);
    pGroupLayout->setContentsMargins(10, 15, 10, 15);
    pGroupLayout->setColumnStretch(1, 1);
    pGroupLayout->setColumnStretch(2, 3);
    pGroupLayout->setSpacing(10);
    if (ParamsModel* paramsM = QVariantPtr<ParamsModel>::asPtr(m_idx.data(ROLE_PARAMS)))
    {
        auto index = paramsM->index(paramsM->indexFromName("background", true), 0);
        if (index.isValid())
        {
            insertRow("background", zeno::ColorVec, index.data(ROLE_PARAM_VALUE), 0, pGroupLayout);
        }
        index = paramsM->index(paramsM->indexFromName("title", true), 0);
        if (index.isValid())
        {
            insertRow("title", zeno::Multiline, index.data(ROLE_PARAM_VALUE), 1, pGroupLayout);
        }
        connect(paramsM, &ParamsModel::dataChanged, this, &ZenoBlackboardPropWidget::onDataChanged);
    }
}

ZenoBlackboardPropWidget::~ZenoBlackboardPropWidget() 
{
}

void ZenoBlackboardPropWidget::onDataChanged(const QModelIndex& topLeft, const QModelIndex& bottomRight, const QVector<int>& roles) {
    if (roles.isEmpty())
        return;
    if (roles[0] == ROLE_PARAM_VALUE ) {
        if (topLeft.data(ROLE_PARAM_NAME).toString() == "title")
        {
            m_pTitle->setText(topLeft.data(ROLE_PARAM_VALUE).toString());
        }
        else if (topLeft.data(ROLE_PARAM_NAME).toString() == "background")
        {
            auto val = topLeft.data(ROLE_PARAM_VALUE).value<UI_VECTYPE>();
            if (val.size() == 3)
            {
                QColor col = QColor::fromRgbF(val[0], val[1], val[2]);
                m_pColor->setStyleSheet(QString("background-color:%1; border:0;").arg(col.name()));
            }
        }
    }

}

void ZenoBlackboardPropWidget::insertRow(const QString &desc, const zeno::ParamControl&ctrl, const QVariant &value, int row,QGridLayout *pGroupLayout) {
    ZTextLabel *pLabel = new ZTextLabel(desc);
    pLabel->setFont(QApplication::font());
    pLabel->setTextColor(QColor(255, 255, 255, 255 * 0.7));
    pLabel->setHoverCursor(Qt::ArrowCursor);

    ZIconLabel *pIcon = new ZIconLabel;
    pIcon->setIcons(ZenoStyle::dpiScaledSize(QSize(24, 24)), ":/icons/parameter_key-frame_idle.svg",
                    ":/icons/parameter_key-frame_hover.svg");
    pGroupLayout->addWidget(pIcon, row, 0, Qt::AlignCenter);

    pGroupLayout->addWidget(pLabel, row, 1, Qt::AlignLeft | Qt::AlignVCenter);

    CallbackCollection cbSet;
    cbSet.cbEditFinished = [=](QVariant newValue) {
        if (ParamsModel* paramsM = QVariantPtr<ParamsModel>::asPtr(m_idx.data(ROLE_PARAMS)))
        {
            auto index = paramsM->index(paramsM->indexFromName(desc, true), 0);
            if (!index.isValid())
                return;
            UiHelper::qIndexSetData(index, newValue, ROLE_PARAM_VALUE);
        }
    };
    zeno::ParamType type = desc == "title" ? zeno::Param_String : zeno::Param_Vec3f;
    QWidget *pControl = zenoui::createWidget(QModelIndex(), value, ctrl, type, cbSet, zeno::ControlProperty());
    pControl->setSizePolicy(QSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred));
    if (desc == "title") {
        m_pTitle = qobject_cast<ZTextEdit *>(pControl);
    } else if (desc == "background") {
        m_pColor = qobject_cast<QPushButton*>(pControl);
    }
    if (pControl)
        pGroupLayout->addWidget(pControl, row, 2, Qt::AlignVCenter);
}