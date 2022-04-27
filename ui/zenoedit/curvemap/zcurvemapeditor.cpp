#include "zcurvemapeditor.h"
#include "curvemapview.h"
#include "ui_zcurvemapeditor.h"
#include <zenoui/style/zenostyle.h>
#include "curvenodeitem.h"
#include "../model/curvemodel.h"
#include "curvesitem.h"
#include <zenoui/util/uihelper.h>


ZCurveMapEditor::ZCurveMapEditor(bool bTimeline, QWidget* parent)
	: QDialog(parent)
    , m_pGroupHdlType(nullptr)
    , m_channelModel(nullptr)
    , m_bTimeline(bTimeline)
{
    initUI();
    initChannelModel();
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

    m_pGroupHdlType = new QButtonGroup(this);
    m_pGroupHdlType->setExclusive(true);
    m_pGroupHdlType->addButton(m_ui->btnFree, HDL_FREE);
    m_pGroupHdlType->addButton(m_ui->btnAligned, HDL_ALIGNED);
    m_pGroupHdlType->addButton(m_ui->btnVector, HDL_VECTOR);
    m_pGroupHdlType->addButton(m_ui->btnAsymmetry, HDL_ASYM);

    //initButtonShadow();
    initStylesheet();
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
    //todo: qt don't offer the implementation of inner shadow, need to implement.
    auto btnList = findChildren<QPushButton*>();
    for (QPushButton* btn : btnList)
    {
        QGraphicsDropShadowEffect *shadowEffect = new QGraphicsDropShadowEffect;
        shadowEffect->setColor(QColor(27, 27, 27, 255 * 0.5));
        shadowEffect->setOffset(0, -1);
        shadowEffect->setBlurRadius(0);
        btn->setGraphicsEffect(shadowEffect);
    }
}

void ZCurveMapEditor::initChannelModel()
{
    //temp code
    if (m_bTimeline)
    {
        m_channelModel = new QStandardItemModel(this);
        QStandardItem* pRootItem = new QStandardItem("Channels");
        m_channelModel->appendRow(pRootItem);
        m_ui->channelView->setModel(m_channelModel);

        m_selection = new QItemSelectionModel(m_channelModel);
    }
    m_ui->channelView->setVisible(m_bTimeline);
}

CurveModel* ZCurveMapEditor::currentModel()
{
    if (m_bTimeline)
    {
        QModelIndex idx = m_ui->channelView->currentIndex();
        QString id = idx.data(Qt::DisplayRole).toString();
        if (m_models.find(id) != m_models.end())
        {
            return m_models[id];
        }
        else
        {
            return nullptr;
        }
    }
    else
    {
        Q_ASSERT(m_models.size() == 1);
        return m_models[0];
    }
}

void ZCurveMapEditor::init(CurveModel* model, bool bTimeFrame)
{
    QString id = model->id();
    m_models.insert(id, model);
    m_ui->gridview->init(model, bTimeFrame);
    if (bTimeFrame)
    {
        QStandardItem* pItem = new QStandardItem(model->id());
        pItem->setCheckable(true);
        pItem->setCheckState(Qt::Checked);
        QStandardItem* pRootItem = m_channelModel->itemFromIndex(m_channelModel->index(0, 0));
        pRootItem->appendRow(pItem);

        CurveGrid *pGrid = m_ui->gridview->gridItem();
        pGrid->setCurvesColor(id, QColor(206, 47, 47));
    }

    CURVE_RANGE range = model->range();
    m_ui->editXFrom->setText(QString::number(range.xFrom));
    m_ui->editXTo->setText(QString::number(range.xTo));
    m_ui->editYFrom->setText(QString::number(range.yFrom));
    m_ui->editYTo->setText(QString::number(range.yTo));

    connect(model, &CurveModel::dataChanged, this, &ZCurveMapEditor::onNodesDataChanged);
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
}

void ZCurveMapEditor::addCurve(CurveModel* model)
{
    QString id = model->id();
    m_models.insert(id, model);
    m_ui->gridview->addCurve(model);

    QStandardItem *pItem = new QStandardItem(model->id());
    pItem->setCheckable(true);
    pItem->setCheckState(Qt::Checked);
    QStandardItem *pRootItem = m_channelModel->itemFromIndex(m_channelModel->index(0, 0));
    pRootItem->appendRow(pItem);

    CurveGrid *pGrid = m_ui->gridview->gridItem();
    pGrid->setCurvesColor(id, QColor(48, 123, 205));

    connect(model, &CurveModel::dataChanged, this, &ZCurveMapEditor::onNodesDataChanged);
    connect(m_channelModel, &QStandardItemModel::dataChanged, this, &ZCurveMapEditor::onChannelModelDataChanged);
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
        Q_ASSERT(idx.isValid());

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
    CurveModel* pModel = qobject_cast<CurveModel*>(sender());// currentModel();
    if (!pModel)
    {
        //temp
        auto lst = m_ui->gridview->getSelectedNodes();
        if (lst.isEmpty())
            return;
        pModel = lst[0]->curves()->model();
        Q_ASSERT(pModel);
    }

    CurveGrid *pGrid = m_ui->gridview->gridItem();
    auto lst = m_ui->gridview->getSelectedNodes();
    bool enableEditor = lst.size() == 1;
    m_ui->editPtX->setEnabled(enableEditor);
    m_ui->editPtY->setEnabled(enableEditor);
    m_ui->editTanLeftX->setEnabled(enableEditor);
    m_ui->editTanLeftY->setEnabled(enableEditor);
    m_ui->editTanRightX->setEnabled(enableEditor);
    m_ui->editTanRightY->setEnabled(enableEditor);
   
    if (lst.size() == 1)
    {
        Q_ASSERT(pGrid && pModel);
        CurveNodeItem *node = lst[0];
        const QModelIndex& idx = node->index();
        QPointF logicPos = pModel->data(idx, ROLE_NODEPOS).toPointF();
        m_ui->editPtX->setText(QString::number(logicPos.x(), 'g', 3));
        m_ui->editPtY->setText(QString::number(logicPos.y(), 'g', 3));

        QPointF leftPos = pModel->data(idx, ROLE_LEFTPOS).toPointF();
        QPointF rightPos = pModel->data(idx, ROLE_RIGHTPOS).toPointF();

        m_ui->editTanLeftX->setText(QString::number(leftPos.x(), 'g', 3));
        m_ui->editTanLeftY->setText(QString::number(leftPos.y() , 'g', 3));
        m_ui->editTanRightX->setText(QString::number(rightPos.x(), 'g', 3));
        m_ui->editTanRightY->setText(QString::number(rightPos.y(), 'g', 3));

        BlockSignalScope scope1(m_ui->btnAsymmetry);
        BlockSignalScope scope2(m_ui->btnAligned);
        BlockSignalScope scope3(m_ui->btnFree);
        BlockSignalScope scope4(m_ui->btnVector);
        BlockSignalScope scope(m_pGroupHdlType);

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
        Q_ASSERT(m_models.find(id) != m_models.end());

        CurveGrid* pGrid = m_ui->gridview->gridItem();
        Q_ASSERT(pGrid);

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