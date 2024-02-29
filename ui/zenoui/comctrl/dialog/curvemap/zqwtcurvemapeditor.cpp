#include "zqwtcurvemapeditor.h"
#include "curvemapview.h"
#include "ui_zqwtcurvemapeditor.h"
#include <zenoui/style/zenostyle.h>
#include "curvenodeitem.h"
#include <zenomodel/include/curvemodel.h>
#include "curvesitem.h"
#include <zenomodel/include/uihelper.h>
#include <zenoui/comctrl/effect/innershadoweffect.h>
#include "zassert.h"
#include <zenomodel/include/igraphsmodel.h>.
#include <zenomodel/include/graphsmanagment.h>
#include "variantptr.h"
#include "plotlayout.h"

ZQwtCurveMapEditor::ZQwtCurveMapEditor(bool bTimeline, QWidget* parent)
	: QDialog(parent)
    , m_pGroupHdlType(nullptr)
    , m_channelModel(nullptr)
    , m_bTimeline(false)
{
    initUI();
    initChannelModel();
    init();
}

ZQwtCurveMapEditor::~ZQwtCurveMapEditor()
{
}

void ZQwtCurveMapEditor::initUI()
{
    m_ui = new Ui::QwtCurveDlg;
    m_ui->setupUi(this);

    setProperty("cssClass", "F-Curve");

    m_ui->btnLoadPreset->setProperty("cssClass", "curve-preset");
    m_ui->btnSavePreset->setProperty("cssClass", "curve-preset");
    m_ui->cbIsTimeline->setProperty("cssClass", "curve-timeline");
    m_ui->btnAddCurve->setProperty("cssClass", "curve-preset");
    m_ui->btnDelCurve->setProperty("cssClass", "curve-preset");

    //todo: move to ui file.
    m_pGroupHdlType = new QButtonGroup(this);
    m_pGroupHdlType->setExclusive(true);
    m_pGroupHdlType->addButton(m_ui->btnFree, HDL_FREE);
    m_pGroupHdlType->addButton(m_ui->btnAligned, HDL_ALIGNED);
    m_pGroupHdlType->addButton(m_ui->btnVector, HDL_VECTOR);
    m_pGroupHdlType->addButton(m_ui->btnAsymmetry, HDL_ASYM);

    m_ui->btnLockX->setIcons(ZenoStyle::dpiScaledSize(QSize(17, 17)),
                             ":/icons/ic_tool_unlock.svg", "", ":/icons/ic_tool_lock.svg");
    m_ui->btnLockY->setIcons(ZenoStyle::dpiScaledSize(QSize(17, 17)),
                             ":/icons/ic_tool_unlock.svg", "", ":/icons/ic_tool_lock.svg");

    initStylesheet();
    initButtonShadow();
    initSize();
    initSignals();
}

void ZQwtCurveMapEditor::initSize()
{
    //qt designer doesn't support dpi scaled size.
    m_ui->editXFrom->setMinimumSize(ZenoStyle::dpiScaledSize(QSize(60, 24)));
    m_ui->editXTo->setMinimumSize(ZenoStyle::dpiScaledSize(QSize(60, 24)));
    m_ui->editYFrom->setMinimumSize(ZenoStyle::dpiScaledSize(QSize(60, 24)));
    m_ui->editYTo->setMinimumSize(ZenoStyle::dpiScaledSize(QSize(60, 24)));

    m_ui->editPtX->setMinimumSize(ZenoStyle::dpiScaledSize(QSize(60, 24)));
    m_ui->editPtY->setMinimumSize(ZenoStyle::dpiScaledSize(QSize(60, 24)));
    m_ui->editFrame->setMinimumSize(ZenoStyle::dpiScaledSize(QSize(60, 24)));
    m_ui->editValue->setMinimumSize(ZenoStyle::dpiScaledSize(QSize(60, 24)));

    m_ui->editTanLeftX->setMinimumSize(ZenoStyle::dpiScaledSize(QSize(65, 24)));
    m_ui->editTanLeftY->setMinimumSize(ZenoStyle::dpiScaledSize(QSize(65, 24)));
    m_ui->editTanRightX->setMinimumSize(ZenoStyle::dpiScaledSize(QSize(65, 24)));
    m_ui->editTanRightY->setMinimumSize(ZenoStyle::dpiScaledSize(QSize(65, 24)));

    QSize size = ZenoStyle::dpiScaledSize(QSize(35, 24));
    m_ui->btnVector->setFixedSize(size);
    m_ui->btnAsymmetry->setFixedSize(size);
    m_ui->btnAligned->setFixedSize(size);
    m_ui->btnFree->setFixedSize(size);

    m_ui->widget->setFixedWidth(ZenoStyle::dpiScaled(180));
    size = QSize(ZenoStyle::dpiScaled(1000), ZenoStyle::dpiScaled(500));
    resize(size);
}

void ZQwtCurveMapEditor::initStylesheet()
{
    auto editors = findChildren<QLineEdit*>(QString(), Qt::FindDirectChildrenOnly);
    for (QLineEdit* pLineEdit : editors)
    {
        pLineEdit->setProperty("cssClass", "FCurve-lineedit");
    }
    m_ui->editPtX->setProperty("cssClass", "FCurve-lineedit");
}

void ZQwtCurveMapEditor::initButtonShadow()
{
    auto btnList = findChildren<QPushButton*>();
    for (QPushButton* btn : btnList)
    {
        InnerShadowEffect *effect = new InnerShadowEffect;
        btn->setGraphicsEffect(effect);
    }
}

void ZQwtCurveMapEditor::initChannelModel()
{
    m_channelModel = new QStandardItemModel(this);
    QStandardItem *pRootItem = new QStandardItem("Channels");
    m_channelModel->appendRow(pRootItem);
    m_ui->channelView->setModel(m_channelModel);
    m_ui->channelView->expandAll();

    connect(m_ui->channelView->selectionModel(), &QItemSelectionModel::currentChanged, this, [=](const QModelIndex& index) {
        QString id = index.data(Qt::DisplayRole).toString();
        emit m_plotLayout->currentModelChanged(id);
    });
    connect(m_channelModel, &QStandardItemModel::dataChanged, this, &ZQwtCurveMapEditor::onChannelModelDataChanged);
    //m_ui->channelView->setVisible(m_ui->cbIsTimeline->isChecked());
}

void ZQwtCurveMapEditor::init()
{
    m_plotLayout = new PlotLayout(m_ui->plotWidget);

    CURVE_RANGE range = { 0,1,0,1 };
    m_ui->editXFrom->setText(QString::number(range.xFrom));
    m_ui->editXFrom->setValidator(new QDoubleValidator);
    m_ui->editXTo->setText(QString::number(range.xTo));
    m_ui->editXTo->setValidator(new QDoubleValidator);
    m_ui->editYFrom->setText(QString::number(range.yFrom));
    m_ui->editYFrom->setValidator(new QDoubleValidator);
    m_ui->editYTo->setText(QString::number(range.yTo));
    m_ui->editYTo->setValidator(new QDoubleValidator);

    connect(m_ui->editPtX, SIGNAL(editingFinished()), this, SLOT(onLineEditFinished()));
    connect(m_ui->editPtY, SIGNAL(editingFinished()), this, SLOT(onLineEditFinished()));
    connect(m_ui->editTanLeftX, SIGNAL(editingFinished()), this, SLOT(onLineEditFinished()));
    connect(m_ui->editTanLeftY, SIGNAL(editingFinished()), this, SLOT(onLineEditFinished()));
    connect(m_ui->editTanRightX, SIGNAL(editingFinished()), this, SLOT(onLineEditFinished()));
    connect(m_ui->editTanRightY, SIGNAL(editingFinished()), this, SLOT(onLineEditFinished()));
    connect(m_plotLayout, &PlotLayout::currentIndexChanged, this, &ZQwtCurveMapEditor::onNodesSelectionChanged);
}

void ZQwtCurveMapEditor::initSignals()
{
    connect(m_pGroupHdlType, SIGNAL(buttonToggled(QAbstractButton *, bool)), this, SLOT(onButtonToggled(QAbstractButton*, bool)));
    //connect(m_ui->gridview, &CurveMapView::nodeItemsSelectionChanged, this, &ZQwtCurveMapEditor::onNodesSelectionChanged);
    //connect(m_ui->gridview, &CurveMapView::frameChanged, this, &ZQwtCurveMapEditor::onFrameChanged);
    connect(m_ui->btnLockX, SIGNAL(toggled(bool)), this, SLOT(onLockBtnToggled(bool)));
    connect(m_ui->btnLockY, SIGNAL(toggled(bool)), this, SLOT(onLockBtnToggled(bool)));
    connect(m_ui->editXFrom, SIGNAL(editingFinished()), this, SLOT(onRangeEdited()));
    connect(m_ui->editXTo, SIGNAL(editingFinished()), this, SLOT(onRangeEdited()));
    connect(m_ui->editYFrom, SIGNAL(editingFinished()), this, SLOT(onRangeEdited()));
    connect(m_ui->editYTo, SIGNAL(editingFinished()), this, SLOT(onRangeEdited()));
    connect(m_ui->cbIsTimeline, SIGNAL(stateChanged(int)), this, SLOT(onCbTimelineChanged(int)));
    connect(m_ui->btnAddCurve, SIGNAL(clicked()), this, SLOT(onAddCurveBtnClicked()));
    connect(m_ui->btnDelCurve, SIGNAL(clicked()), this, SLOT(onDelCurveBtnClicked()));
}

void ZQwtCurveMapEditor::addCurves(const CURVES_DATA& curves)
{
    for (QString key : curves.keys())
    {
        CURVE_DATA curve = curves[key];
        CurveModel* model = new CurveModel(key, curve.rg, this);
        model->initItems(curve);
        model->setVisible(curve.visible);
        model->setTimeline(curve.timeline);
        addCurve(model);
    }
}

void ZQwtCurveMapEditor::setCurrentChannel(const QString& id)
{
    QStandardItem* pRootItem = m_channelModel->itemFromIndex(m_channelModel->index(0, 0));
    for (int i = 0; i < pRootItem->rowCount(); i++)
    {
        if (pRootItem->child(i, 0)->data(Qt::DisplayRole).toString() == id)
        {
            m_ui->channelView->setCurrentIndex(pRootItem->child(i, 0)->index());
            break;
        }
    }
}

void ZQwtCurveMapEditor::addCurve(CurveModel *model)
{
    m_plotLayout->addCurve(model);
    m_ui->cbIsTimeline->setChecked(model->isTimeline());

    CURVE_RANGE range = model->range();
    m_ui->editXFrom->setText(QString::number(range.xFrom));
    m_ui->editXTo->setText(QString::number(range.xTo));
    m_ui->editYFrom->setText(QString::number(range.yFrom));
    m_ui->editYTo->setText(QString::number(range.yTo));
  
    m_bate_rows.push_back(model);

    QStandardItem *pItem = new QStandardItem(model->id());
    pItem->setCheckable(true);
    pItem->setCheckState(model->getVisible() ? Qt::Checked : Qt::Unchecked);
    QStandardItem *pRootItem = m_channelModel->itemFromIndex(m_channelModel->index(0, 0));
    if (pRootItem->rowCount() == 0)
    {
        pRootItem->appendRow(pItem);
    }
    else {
        int i = 0;
        while (pRootItem->child(i, 0) != NULL && model->id() > pRootItem->child(i, 0)->data(Qt::DisplayRole).toString())
        {
            i++;
        }
        pRootItem->insertRow(i, pItem);
    }
    setCurrentChannel(model->id());
    connect(model, &CurveModel::dataChanged, this, &ZQwtCurveMapEditor::onNodesDataChanged);
}

void ZQwtCurveMapEditor::onRangeEdited()
{
    CURVE_RANGE newRg = {m_ui->editXFrom->text().toDouble(), m_ui->editXTo->text().toDouble(),
                         m_ui->editYFrom->text().toDouble(), m_ui->editYTo->text().toDouble()};
    const auto models = m_plotLayout->curveModels();
    if (models.isEmpty())
        return;
        
    CURVE_RANGE rg = models.first()->range();
    if (rg.xFrom != newRg.xFrom || rg.xTo != newRg.xTo || rg.yFrom != newRg.yFrom || rg.yTo != newRg.yTo)
    {
        m_plotLayout->updateRange(newRg);
    }
}

void ZQwtCurveMapEditor::onCbTimelineChanged(int state)
{
    //if (state == Qt::Checked) {
    //    m_ui->gridview->setChartType(true);
    //} else if (state == Qt::Unchecked) {
    //    m_ui->gridview->setChartType(false);
    //}
    for (CurveModel* model : m_plotLayout->curveModels())
    {
        model->setTimeline(state == Qt::Checked);
    }
}

void ZQwtCurveMapEditor::onAddCurveBtnClicked() {
    QStandardItem * pRootItem = m_channelModel->itemFromIndex(m_channelModel->index(0, 0));
    if (pRootItem->rowCount() != 3)
    {
        CurveModel *newCurve = curve_util::deflModel(this);
        const CurveModel* currModel = m_plotLayout->currentModel();
        if (currModel)
        {
            newCurve->resetRange(currModel->range());
        }
        if (pRootItem->child(0, 0) == NULL || pRootItem->child(0, 0)->data(Qt::DisplayRole) != "x")
        {
            newCurve->setId("x");
            newCurve->setData(newCurve->index(0, 0), QVariant::fromValue(QPointF(0, 0)), ROLE_NODEPOS);
            newCurve->setData(newCurve->index(1, 0), QVariant::fromValue(QPointF(1, 1)), ROLE_NODEPOS);
            addCurve(newCurve);
        } else if (pRootItem->child(1, 0) == NULL || pRootItem->child(1, 0)->data(Qt::DisplayRole) != "y")
        {
            newCurve->setData(newCurve->index(0, 0), QVariant::fromValue(QPointF(0, 0.5)), ROLE_NODEPOS);
            newCurve->setData(newCurve->index(1, 0), QVariant::fromValue(QPointF(1, 0.5)), ROLE_NODEPOS);
            newCurve->setId("y");
            addCurve(newCurve);
        } else {
            newCurve->setData(newCurve->index(0, 0), QVariant::fromValue(QPointF(0, 1)), ROLE_NODEPOS);
            newCurve->setData(newCurve->index(1, 0), QVariant::fromValue(QPointF(1, 0)), ROLE_NODEPOS);
            newCurve->setId("z");
            addCurve(newCurve);
        }
    }
}

void ZQwtCurveMapEditor::onDelCurveBtnClicked() {
    QModelIndexList lst = m_ui->channelView->selectionModel()->selectedIndexes();
    if (lst.size() != 0 && lst[0] != m_channelModel->index(0, 0))
    {
        QStandardItem *pRootItem = m_channelModel->itemFromIndex(m_channelModel->index(0, 0));
        QString curveName = lst[0].data(Qt::DisplayRole).toString();
        pRootItem->removeRow(lst[0].row());
        const auto& models = getModel();
        if (models.contains(curveName))
            m_bate_rows.erase(std::find(m_bate_rows.begin(), m_bate_rows.end(), models[curveName]));
        m_plotLayout->deleteCurve(curveName);
    }
}

int ZQwtCurveMapEditor::curveCount() const {
    return (int)m_bate_rows.size();
}

CurveModel *ZQwtCurveMapEditor::getCurve(int i) const {
    return m_bate_rows.at(i);
}

CURVES_MODEL ZQwtCurveMapEditor::getModel() const {
    return m_plotLayout->curveModels();
}

CURVES_DATA ZQwtCurveMapEditor::curves() const
{
    CURVES_DATA curves;
    const auto& models = getModel();
    for (QString key : models.keys())
    {
        CURVE_DATA data = models[key]->getItems();
        data.visible = models[key]->getVisible();
        data.timeline = models[key]->isTimeline();
        curves.insert(key, data);
    }
    return curves;
}

void ZQwtCurveMapEditor::onButtonToggled(QAbstractButton* btn, bool bToggled)
{
    const auto& index = m_plotLayout->currentIndex();
    QAbstractItemModel* pModel = const_cast<QAbstractItemModel*>(index.model());
    if (!bToggled || !pModel)
        return;

    if (index.isValid())
    {
        if (btn == m_ui->btnVector)
        {
            pModel->setData(index, HDL_VECTOR, ROLE_TYPE);
        }
        else if (btn == m_ui->btnAligned)
        {
            pModel->setData(index, HDL_ALIGNED, ROLE_TYPE);
        }
        else if (btn == m_ui->btnAsymmetry)
        {
            pModel->setData(index, HDL_ASYM, ROLE_TYPE);
        }
        else if (btn == m_ui->btnFree)
        {
            pModel->setData(index, HDL_FREE, ROLE_TYPE);
        }
    }
}

void ZQwtCurveMapEditor::onLineEditFinished()
{
    const auto& index = m_plotLayout->currentIndex();
    QAbstractItemModel* pModel = const_cast<QAbstractItemModel*>(index.model());
    if (!pModel)
        return;

    QObject *pEdit = sender();
    if (pEdit == m_ui->editPtX) {
    
    }

    if (index.isValid())
    {
        QPointF logicPos = QPointF(m_ui->editPtX->text().toFloat(), m_ui->editPtY->text().toFloat());
        qreal leftX = m_ui->editTanLeftX->text().toFloat();
        qreal leftY = m_ui->editTanLeftY->text().toFloat();
        qreal rightX = m_ui->editTanRightX->text().toFloat();
        qreal rightY = m_ui->editTanRightY->text().toFloat();
        QPointF leftHdlLogic = logicPos + QPointF(leftX, leftY);
        QPointF rightHdlLogic = logicPos + QPointF(rightX, rightY);

        pModel->setData(index, logicPos, ROLE_NODEPOS);
        pModel->setData(index, QPointF(leftX, leftY), ROLE_LEFTPOS);
        pModel->setData(index, QPointF(rightX, rightY), ROLE_RIGHTPOS);
    }
}

void ZQwtCurveMapEditor::onNodesDataChanged(const QModelIndex& topLeft, const QModelIndex& bottomRight, const QVector<int>& roles)
{
    if (!topLeft.isValid() || roles.isEmpty())
        return;
    onNodesSelectionChanged();
}

void ZQwtCurveMapEditor::onLockBtnToggled(bool bToggle)
{
    if (sender() != m_ui->btnLockX && sender() != m_ui->btnLockY)
        return;

    const auto& index = m_plotLayout->currentIndex();
    if (index.isValid())
    {
        QAbstractItemModel* pModel = const_cast<QAbstractItemModel*>(index.model());
        if (pModel)
        {
            pModel->setData(index, bToggle, sender() == m_ui->btnLockX ? ROLE_LOCKX : ROLE_LOCKY);
        }
    }
}

void ZQwtCurveMapEditor::onNodesSelectionChanged()
{
    const auto& index = m_plotLayout->currentIndex();
    bool enableEditor = index.isValid();
    m_ui->editPtX->setEnabled(enableEditor);
    m_ui->editPtX->setText("");
    m_ui->editPtY->setEnabled(enableEditor);
    m_ui->editPtY->setText("");
    m_ui->editTanLeftX->setEnabled(enableEditor);
    m_ui->editTanLeftX->setText("");
    m_ui->editTanLeftY->setEnabled(enableEditor);
    m_ui->editTanLeftY->setText("");
    m_ui->editTanRightX->setEnabled(enableEditor);
    m_ui->editTanRightX->setText("");
    m_ui->editTanRightY->setEnabled(enableEditor);
    m_ui->editTanRightY->setText("");
    m_ui->btnLockX->setEnabled(enableEditor);
    m_ui->btnLockX->toggle(false);
    m_ui->btnLockX->setText("");
    m_ui->btnLockY->setEnabled(enableEditor);
    m_ui->btnLockY->toggle(false);
    m_ui->btnLockY->setText("");

    if (enableEditor)
    {
        QPointF logicPos = index.data(ROLE_NODEPOS).toPointF();
        m_ui->editPtX->setText(QString::number(logicPos.x(), 'g', 3));
        m_ui->editPtY->setText(QString::number(logicPos.y(), 'g', 3));

        QPointF leftPos = index.data(ROLE_LEFTPOS).toPointF();
        QPointF rightPos = index.data(ROLE_RIGHTPOS).toPointF();

        bool bLockX = index.data(ROLE_LOCKX).toBool();
        bool bLockY = index.data(ROLE_LOCKY).toBool();

        m_ui->editTanLeftX->setText(QString::number(leftPos.x(), 'g', 3));
        m_ui->editTanLeftY->setText(QString::number(leftPos.y(), 'g', 3));
        m_ui->editTanRightX->setText(QString::number(rightPos.x(), 'g', 3));
        m_ui->editTanRightY->setText(QString::number(rightPos.y(), 'g', 3));

        BlockSignalScope scope1(m_ui->btnAsymmetry);
        BlockSignalScope scope2(m_ui->btnAligned);
        BlockSignalScope scope3(m_ui->btnFree);
        BlockSignalScope scope4(m_ui->btnVector);
        BlockSignalScope scope(m_pGroupHdlType);
        BlockSignalScope scope_(m_ui->btnLockX);
        BlockSignalScope scope__(m_ui->btnLockY);

        m_ui->btnLockX->toggle(bLockX);
        m_ui->btnLockY->toggle(bLockY);

        switch (index.data(ROLE_TYPE).toInt())
        {
        case HDL_ASYM:
        {
            m_ui->btnAsymmetry->setChecked(true);
            break;
        }
        case HDL_ALIGNED:
        {
            m_ui->btnAligned->setChecked(true);
            break;
        }
        case HDL_FREE:
        {
            m_ui->btnFree->setChecked(true);
            break;
        }
        case HDL_VECTOR:
        {
            m_ui->btnVector->setChecked(true);
            break;
        }
        }
    }
}

void ZQwtCurveMapEditor::onChannelModelDataChanged(const QModelIndex &topLeft, const QModelIndex &bottomRight, const QVector<int> &roles)
{
    if (!topLeft.isValid() || roles.isEmpty())
        return;

    QString id = topLeft.data(Qt::DisplayRole).toString();
    if (roles[0] == Qt::CheckStateRole)
    {
        Qt::CheckState state = topLeft.data(Qt::CheckStateRole).value<Qt::CheckState>();
        bool bVisible = (state == Qt::Checked);
        m_plotLayout->setVisible(id, bVisible);
    }
}

void ZQwtCurveMapEditor::onFrameChanged(qreal frame)
{
    m_ui->editFrame->setText(QString::number(frame));
}
