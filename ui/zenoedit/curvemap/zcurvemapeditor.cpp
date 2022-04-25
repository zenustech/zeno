#include "zcurvemapeditor.h"
#include "curvemapview.h"
#include "ui_zcurvemapeditor.h"
#include <zenoui/style/zenostyle.h>
#include "curvenodeitem.h"
#include "../model/curvemodel.h"
#include <zenoui/util/uihelper.h>


ZCurveMapEditor::ZCurveMapEditor(QWidget* parent)
	: QDialog(parent)
    , m_pGroupHdlType(nullptr)
    , m_model(nullptr)
{
    initUI();
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

void ZCurveMapEditor::init(CurveModel* model)
{
    m_model = model;
    m_ui->gridview->init(model);
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
}

void ZCurveMapEditor::onButtonToggled(QAbstractButton* btn, bool bToggled)
{
    if (!bToggled)
        return;

    auto lst = m_ui->gridview->getSelectedNodes();
    if (lst.size() == 1)
    {
        CurveNodeItem* node = lst[0];
        QModelIndex idx = node->index();
        Q_ASSERT(idx.isValid());
        //todo: when change type from vector to other, should init a offset
        // otherwise it's hard to control them.

        if (btn == m_ui->btnVector)
        {
            m_model->setData(idx, HDL_VECTOR, ROLE_TYPE);

            //todo: model sync to items.
            //m_model->setData(idx, 0, ROLE_LEFTPOS);
            //m_model->setData(idx, 0, ROLE_RIGHTPOS);
            //temp code:
            if (node->leftHandle()) {
                node->leftHandle()->setPos(QPointF(0, 0));
                node->leftHandle()->toggle(false);
            }
            if (node->rightHandle()) {
                node->rightHandle()->setPos(QPointF(0, 0));
                node->rightHandle()->toggle(false);
            }
        }
        else
        {
            bool bVector = m_model->data(idx, ROLE_TYPE).toInt() == HDL_VECTOR;
            if (btn == m_ui->btnAligned)
            {
                m_model->setData(idx, HDL_ALIGNED, ROLE_TYPE);
            }
            else if (btn == m_ui->btnAsymmetry)
            {
                m_model->setData(idx, HDL_ASYM, ROLE_TYPE);
            }
            else if (btn == m_ui->btnFree)
            {
                m_model->setData(idx, HDL_FREE, ROLE_TYPE);
            }
            //show handles.
            if (bVector)
            {
                if (node->leftHandle())
                {
                    node->leftHandle()->setPos(QPointF(-0.1, -0.1));
                }
                if (node->rightHandle())
                {
                    node->rightHandle()->setPos(QPointF(0.1, 0.1));
                }
            }
        }
    }
}

void ZCurveMapEditor::onLineEditFinished()
{
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
        m_model->setData(idx, logicPos, ROLE_NODEPOS);
        m_model->setData(idx, QPointF(leftX, leftY), ROLE_LEFTPOS);
        m_model->setData(idx, QPointF(rightX, rightY), ROLE_RIGHTPOS);
    }
}

void ZCurveMapEditor::onNodesDataChanged()
{
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
        Q_ASSERT(pGrid && m_model);
        CurveNodeItem *node = lst[0];
        const QModelIndex& idx = node->index();
        QPointF logicPos = m_model->data(idx, ROLE_NODEPOS).toPointF();
        m_ui->editPtX->setText(QString::number(logicPos.x(), 'g', 3));
        m_ui->editPtY->setText(QString::number(logicPos.y(), 'g', 3));

        QPointF leftPos = m_model->data(idx, ROLE_LEFTPOS).toPointF();
        QPointF rightPos = m_model->data(idx, ROLE_RIGHTPOS).toPointF();

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