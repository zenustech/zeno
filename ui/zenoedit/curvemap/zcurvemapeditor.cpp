#include "zcurvemapeditor.h"
#include "curvemapview.h"
#include "ui_zcurvemapeditor.h"


ZCurveMapEditor::ZCurveMapEditor(QWidget* parent)
	: QDialog(parent)
	, m_view(nullptr)
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
    m_ui->btnFree->setProperty("cssClass", "shadowButton");
    m_ui->btnVector->setProperty("cssClass", "shadowButton");
    m_ui->btnAligned->setProperty("cssClass", "shadowButton");
    m_ui->btnAsymmetry->setProperty("cssClass", "shadowButton");
    m_ui->btnLoadPreset->setProperty("cssClass", "shadowButton");
    m_ui->btnSavePreset->setProperty("cssClass", "shadowButton");
}

void ZCurveMapEditor::init(CURVE_RANGE range, const QVector<QPointF>& pts, const QVector<QPointF>& handlers)
{
	m_ui->gridview->init(range, pts, handlers);
}