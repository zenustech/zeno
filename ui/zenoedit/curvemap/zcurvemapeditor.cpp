#include "zcurvemapeditor.h"
#include "curvemapview.h"
#include "ui_zcurvemapeditor.h"
#include <zenoui/style/zenostyle.h>
#include "curvenodeitem.h"


ZCurveMapEditor::ZCurveMapEditor(QWidget* parent)
	: QDialog(parent)
    , m_pGroupHdlType(nullptr)
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
    m_pGroupHdlType->addButton(m_ui->btnFree, curve_util::HDL_FREE);
    m_pGroupHdlType->addButton(m_ui->btnAligned, curve_util::HDL_ALIGNED);
    m_pGroupHdlType->addButton(m_ui->btnVector, curve_util::HDL_VECTOR);
    m_pGroupHdlType->addButton(m_ui->btnAsymmetry, curve_util::HDL_ASYM);

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

void ZCurveMapEditor::init(CURVE_RANGE range, const QVector<QPointF>& pts, const QVector<QPointF>& handlers)
{
	m_ui->gridview->init(range, pts, handlers);
    m_ui->editXFrom->setText(QString::number(range.xFrom));
    m_ui->editXTo->setText(QString::number(range.xTo));
    m_ui->editYFrom->setText(QString::number(range.yFrom));
    m_ui->editYTo->setText(QString::number(range.yTo));

    connect(m_ui->gridview->gridItem(), SIGNAL(nodesDataChanged()), this, SLOT(onNodesDataChanged()));
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
    auto lst = m_ui->gridview->getSelectedNodes();

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

        node->setPos(nodeScenePos);
        if (node->leftHandle())
            node->leftHandle()->setPos(leftHdlOffset);
        if (node->rightHandle())
            node->rightHandle()->setPos(rightHdlOffset);
    }
}

void ZCurveMapEditor::onNodesDataChanged()
{
    CurveGrid *pGrid = m_ui->gridview->gridItem();
    auto lst = m_ui->gridview->getSelectedNodes();
    if (lst.isEmpty())
    {
        m_ui->editPtX->setText("");
        m_ui->editPtY->setText("");
        m_ui->editTanLeftX->setText("");
        m_ui->editTanLeftY->setText("");
        m_ui->editTanRightX->setText("");
        m_ui->editTanRightY->setText("");
    } 
    else if (lst.size() > 1)
    {
        //todo
        m_ui->editPtX->setText("");
        m_ui->editPtY->setText("");
        m_ui->editTanLeftX->setText("");
        m_ui->editTanLeftY->setText("");
        m_ui->editTanRightX->setText("");
        m_ui->editTanRightY->setText("");
    }
    else
    {
        Q_ASSERT(pGrid);
        CurveNodeItem *node = lst[0];
        QPointF scenePos = node->scenePos();
        QPointF logicPos = pGrid->sceneToLogic(scenePos);
        m_ui->editPtX->setText(QString::number(logicPos.x(), 'g', 3));
        m_ui->editPtY->setText(QString::number(logicPos.y(), 'g', 3));

        QPointF leftHdlLogicPos = pGrid->sceneToLogic(node->leftHandlePos());
        QPointF rightHdlLogicPos = pGrid->sceneToLogic(node->rightHandlePos());
        qreal xLeftOffset = leftHdlLogicPos.x() - logicPos.x();
        qreal yLeftOffset = leftHdlLogicPos.y() - logicPos.y();
        qreal xRightOffset = rightHdlLogicPos.x() - logicPos.x();
        qreal yRightOffset = rightHdlLogicPos.y() - logicPos.y();

        m_ui->editTanLeftX->setText(QString::number(xLeftOffset, 'g', 3));
        m_ui->editTanLeftY->setText(QString::number(yLeftOffset, 'g', 3));
        m_ui->editTanRightX->setText(QString::number(xRightOffset, 'g', 3));
        m_ui->editTanRightY->setText(QString::number(yRightOffset, 'g', 3));
    }
}

void ZCurveMapEditor::onNodesSelectionChanged(QList<CurveNodeItem*> lst)
{
    onNodesDataChanged();
}