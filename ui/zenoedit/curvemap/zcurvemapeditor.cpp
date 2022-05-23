#include "zcurvemapeditor.h"
#include "curvemapview.h"
#include "ui_zcurvemapeditor.h"
#include <zenoui/style/zenostyle.h>
#include "curvenodeitem.h"
#include <zenoui/model/curvemodel.h>
#include "curvesitem.h"
#include <zenoui/util/uihelper.h>
#include <zenoui/comctrl/effect/innershadoweffect.h>
#include "util/log.h"


ZCurveMapEditor::ZCurveMapEditor(bool bTimeline, QWidget* parent)
	: QDialog(parent)
    , m_pGroupHdlType(nullptr)
    , m_channelModel(nullptr)
    , m_bTimeline(false)
{
    initUI();
    initChannelModel();
    init();
}

ZCurveMapEditor::~ZCurveMapEditor()
{
}

void ZCurveMapEditor::initUI()
{
    m_ui = new Ui::FCurveDlg;
    m_ui->setupUi(this);

    setProperty("cssClass", "F-Curve");

    m_ui->btnLoadPreset->setProperty("cssClass", "curve-preset");
    m_ui->btnSavePreset->setProperty("cssClass", "curve-preset");

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
    //initSize();
    initSignals();
}

void ZCurveMapEditor::initSize()
{
    //qt designer doesn't support dpi scaled size.
    m_ui->editXFrom->setFixedSize(ZenoStyle::dpiScaledSize(QSize(36, 20)));
    m_ui->editXTo->setFixedSize(ZenoStyle::dpiScaledSize(QSize(36, 20)));
    m_ui->editYFrom->setFixedSize(ZenoStyle::dpiScaledSize(QSize(36, 20)));
    m_ui->editYTo->setFixedSize(ZenoStyle::dpiScaledSize(QSize(36, 20)));

    m_ui->editPtX->setFixedSize(ZenoStyle::dpiScaledSize(QSize(60, 20)));
    m_ui->editPtY->setFixedSize(ZenoStyle::dpiScaledSize(QSize(60, 20)));

    m_ui->editTanLeftX->setFixedSize(ZenoStyle::dpiScaledSize(QSize(60, 20)));
    m_ui->editTanLeftY->setFixedSize(ZenoStyle::dpiScaledSize(QSize(60, 20)));
    m_ui->editTanRightX->setFixedSize(ZenoStyle::dpiScaledSize(QSize(60, 20)));
    m_ui->editTanRightY->setFixedSize(ZenoStyle::dpiScaledSize(QSize(60, 20)));
}

void ZCurveMapEditor::initStylesheet()
{
    auto editors = findChildren<QLineEdit*>(QString(), Qt::FindDirectChildrenOnly);
    for (QLineEdit* pLineEdit : editors)
    {
        pLineEdit->setProperty("cssClass", "FCurve-lineedit");
    }
    m_ui->editPtX->setProperty("cssClass", "FCurve-lineedit");
}

void ZCurveMapEditor::initButtonShadow()
{
    auto btnList = findChildren<QPushButton*>();
    for (QPushButton* btn : btnList)
    {
        InnerShadowEffect *effect = new InnerShadowEffect;
        btn->setGraphicsEffect(effect);
    }
}

void ZCurveMapEditor::initChannelModel()
{
    m_channelModel = new QStandardItemModel(this);
    QStandardItem* pRootItem = new QStandardItem("Channels");
    m_channelModel->appendRow(pRootItem);
    m_ui->channelView->setModel(m_channelModel);
    m_ui->channelView->expandAll();

    m_selection = new QItemSelectionModel(m_channelModel);
    m_ui->channelView->setVisible(m_ui->cbIsTimeline->isChecked());
}

CurveModel* ZCurveMapEditor::currentModel()
{
    auto lst = m_ui->gridview->getSelectedNodes();
    if (lst.size() == 0)
        return nullptr;
    return lst[0]->curves()->model();
}

void ZCurveMapEditor::init()
{
    m_ui->gridview->init(m_ui->cbIsTimeline->isChecked());

    CURVE_RANGE range = m_ui->gridview->range();
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
}

void ZCurveMapEditor::initSignals()
{
    connect(m_pGroupHdlType, SIGNAL(buttonToggled(QAbstractButton *, bool)), this, SLOT(onButtonToggled(QAbstractButton*, bool)));
    connect(m_ui->gridview, &CurveMapView::nodeItemsSelectionChanged, this, &ZCurveMapEditor::onNodesSelectionChanged);
    connect(m_ui->gridview, &CurveMapView::frameChanged, this, &ZCurveMapEditor::onFrameChanged);
    connect(m_ui->btnLockX, SIGNAL(toggled(bool)), this, SLOT(onLockBtnToggled(bool)));
    connect(m_ui->btnLockY, SIGNAL(toggled(bool)), this, SLOT(onLockBtnToggled(bool)));
    connect(m_ui->editXFrom, SIGNAL(editingFinished()), this, SLOT(onRangeEdited()));
    connect(m_ui->editXTo, SIGNAL(editingFinished()), this, SLOT(onRangeEdited()));
    connect(m_ui->editYFrom, SIGNAL(editingFinished()), this, SLOT(onRangeEdited()));
    connect(m_ui->editYTo, SIGNAL(editingFinished()), this, SLOT(onRangeEdited()));
    connect(m_ui->cbIsTimeline, SIGNAL(stateChanged(int)), this, SLOT(onCbTimelineChanged(int)));
}

void ZCurveMapEditor::addCurve(CurveModel* model)
{
    static const QColor preset[] = {"#CE2F2F", "#2FCD5F", "#307BCD"};

    QString id = model->id();
    m_models.insert(id, model);
    m_ui->gridview->addCurve(model);

    m_ui->cbIsTimeline->setChecked(model->isTimeline());

    CURVE_RANGE range = m_ui->gridview->range();
    m_ui->editXFrom->setText(QString::number(range.xFrom));
    m_ui->editXTo->setText(QString::number(range.xTo));
    m_ui->editYFrom->setText(QString::number(range.yFrom));
    m_ui->editYTo->setText(QString::number(range.yTo));

    QStandardItem *pItem = new QStandardItem(model->id());
    pItem->setCheckable(true);
    pItem->setCheckState(Qt::Checked);
    QStandardItem *pRootItem = m_channelModel->itemFromIndex(m_channelModel->index(0, 0));

    int n = pRootItem->rowCount();
    QColor curveClr;
    if (n < sizeof(preset) / sizeof(QColor))
    {
        curveClr = preset[n];
    }
    else
    {
        curveClr = QColor(77, 77, 77);
    }

    pRootItem->appendRow(pItem);
    m_bate_rows.push_back(model);
    CurveGrid *pGrid = m_ui->gridview->gridItem();
    pGrid->setCurvesColor(id, curveClr);

    connect(model, &CurveModel::dataChanged, this, &ZCurveMapEditor::onNodesDataChanged);
    connect(m_channelModel, &QStandardItemModel::dataChanged, this, &ZCurveMapEditor::onChannelModelDataChanged);
}

void ZCurveMapEditor::onRangeEdited()
{
    CURVE_RANGE newRg = {m_ui->editXFrom->text().toDouble(), m_ui->editXTo->text().toDouble(),
                         m_ui->editYFrom->text().toDouble(), m_ui->editYTo->text().toDouble()};
    CURVE_RANGE rg = m_ui->gridview->range();
    if (rg.xFrom != newRg.xFrom || rg.xTo != newRg.xTo || rg.yFrom != newRg.yFrom || rg.yTo != newRg.yTo)
    {
        m_ui->gridview->resetRange(newRg);
    }
}

void ZCurveMapEditor::onCbTimelineChanged(int state)
{
    if (state == Qt::Checked) {
        m_ui->gridview->setChartType(true);
    } else if (state == Qt::Unchecked) {
        m_ui->gridview->setChartType(false);
    }
    for (CurveModel* model : m_models)
    {
        model->setTimeline(state == Qt::Checked);
    }
}

int ZCurveMapEditor::curveCount() const {
    return (int)m_bate_rows.size();
}

CurveModel *ZCurveMapEditor::getCurve(int i) const {
    return m_bate_rows.at(i);
}

void ZCurveMapEditor::onButtonToggled(QAbstractButton* btn, bool bToggled)
{
    CurveModel *pModel = currentModel();
    if (!bToggled || !pModel)
        return;

    auto lst = m_ui->gridview->getSelectedNodes();
    if (lst.size() == 1)
    {
        CurveNodeItem* node = lst[0];
        QModelIndex idx = node->index();
        ZASSERT_EXIT(idx.isValid());

        if (btn == m_ui->btnVector)
        {
            pModel->setData(idx, HDL_VECTOR, ROLE_TYPE);
        }
        else if (btn == m_ui->btnAligned)
        {
            pModel->setData(idx, HDL_ALIGNED, ROLE_TYPE);
        }
        else if (btn == m_ui->btnAsymmetry)
        {
            pModel->setData(idx, HDL_ASYM, ROLE_TYPE);
        }
        else if (btn == m_ui->btnFree)
        {
            pModel->setData(idx, HDL_FREE, ROLE_TYPE);
        }
    }
}

void ZCurveMapEditor::onLineEditFinished()
{
    CurveModel* pModel = currentModel();
    if (!pModel)
        return;

    QObject *pEdit = sender();
    if (pEdit == m_ui->editPtX) {
    
    }

    CurveGrid *pGrid = m_ui->gridview->gridItem();
    auto lst = m_ui->gridview->getSelectedNodes();
    if (lst.size() == 1)
    {
        CurveNodeItem* node = lst[0];
        QPointF logicPos = QPointF(m_ui->editPtX->text().toFloat(), m_ui->editPtY->text().toFloat());
        qreal leftX = m_ui->editTanLeftX->text().toFloat();
        qreal leftY = m_ui->editTanLeftY->text().toFloat();
        qreal rightX = m_ui->editTanRightX->text().toFloat();
        qreal rightY = m_ui->editTanRightY->text().toFloat();
        QPointF leftHdlLogic = logicPos + QPointF(leftX, leftY);
        QPointF rightHdlLogic = logicPos + QPointF(rightX, rightY);

        QPointF nodeScenePos = pGrid->logicToScene(logicPos);
        QPointF leftHdlScene = pGrid->logicToScene(leftHdlLogic);
        QPointF rightHdlScene = pGrid->logicToScene(rightHdlLogic);
        QPointF leftHdlOffset = leftHdlScene - nodeScenePos;
        QPointF rightHdlOffset = rightHdlScene - nodeScenePos;

        const QModelIndex& idx = node->index();
        pModel->setData(idx, logicPos, ROLE_NODEPOS);
        pModel->setData(idx, QPointF(leftX, leftY), ROLE_LEFTPOS);
        pModel->setData(idx, QPointF(rightX, rightY), ROLE_RIGHTPOS);
    }
}

void ZCurveMapEditor::onNodesDataChanged()
{
    CurveGrid *pGrid = m_ui->gridview->gridItem();
    auto lst = m_ui->gridview->getSelectedNodes();
    bool enableEditor = lst.size() == 1;
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
   
    if (lst.size() == 1)
    {
        ZASSERT_EXIT(pGrid);
        CurveNodeItem *node = lst[0];
        const QModelIndex& idx = node->index();
        QPointF logicPos = idx.data(ROLE_NODEPOS).toPointF();
        m_ui->editPtX->setText(QString::number(logicPos.x(), 'g', 3));
        m_ui->editPtY->setText(QString::number(logicPos.y(), 'g', 3));

        QPointF leftPos = idx.data(ROLE_LEFTPOS).toPointF();
        QPointF rightPos = idx.data(ROLE_RIGHTPOS).toPointF();

        bool bLockX = idx.data(ROLE_LOCKX).toBool();
        bool bLockY = idx.data(ROLE_LOCKY).toBool();

        m_ui->editTanLeftX->setText(QString::number(leftPos.x(), 'g', 3));
        m_ui->editTanLeftY->setText(QString::number(leftPos.y() , 'g', 3));
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

        switch (idx.data(ROLE_TYPE).toInt())
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

void ZCurveMapEditor::onLockBtnToggled(bool bToggle)
{
    if (sender() != m_ui->btnLockX && sender() != m_ui->btnLockY)
        return;

    auto lst = m_ui->gridview->getSelectedNodes();
    if (lst.size() == 1)
    {
        CurveNodeItem *node = lst[0];
        QModelIndex idx = node->index();
        ZASSERT_EXIT(idx.isValid());
        CurveModel *pModel = currentModel();
        if (pModel)
        {
            pModel->setData(idx, bToggle, sender() == m_ui->btnLockX ? ROLE_LOCKX : ROLE_LOCKY);
        }
    }
}

void ZCurveMapEditor::onNodesSelectionChanged(QList<CurveNodeItem*> lst)
{
    onNodesDataChanged();
}

void ZCurveMapEditor::onChannelModelDataChanged(const QModelIndex &topLeft, const QModelIndex &bottomRight, const QVector<int> &roles)
{
    if (!topLeft.isValid() || roles.isEmpty())
        return;

    QString id = topLeft.data(Qt::DisplayRole).toString();
    if (roles[0] == Qt::CheckStateRole)
    {
        ZASSERT_EXIT(m_models.find(id) != m_models.end());

        CurveGrid* pGrid = m_ui->gridview->gridItem();
        ZASSERT_EXIT(pGrid);

        Qt::CheckState state = topLeft.data(Qt::CheckStateRole).value<Qt::CheckState>();
        if (state == Qt::Checked)
        {
            pGrid->setCurvesVisible(id, true);
        }
        else if (state == Qt::Unchecked)
        {
            pGrid->setCurvesVisible(id, false);
        }
    }
}

void ZCurveMapEditor::onFrameChanged(qreal frame)
{
    m_ui->editFrame->setText(QString::number(frame));
}
