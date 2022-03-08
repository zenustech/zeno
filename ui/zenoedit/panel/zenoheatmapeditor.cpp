#include "ui_zenoheatmapeditor.h"
#include "zenoheatmapeditor.h"
#include <QtWidgets>


ZenoHeatMapEditor::ZenoHeatMapEditor(QWidget* parent)
	: QWidget(parent)
{
	setWindowFlags(Qt::Dialog | Qt::MSWindowsFixedSizeDialogHint);
	m_ui = new Ui::HeatMapEditor;
	m_ui->setupUi(this);
	init();
	initSignals();
	installFilters();
}

void ZenoHeatMapEditor::init()
{
	initRamps();
	m_ui->cbPreset->addItems({"BlackBody", "Grayscale", "InfraRed", "TwoTone", "WhiteToRed"});
	m_ui->hueSlider;
}

void ZenoHeatMapEditor::installFilters()
{
	//m_ui->rampSelector2->installEventFilter(this);
}

void ZenoHeatMapEditor::initSignals()
{
}

void ZenoHeatMapEditor::initRamps()
{
	m_ui->rampSelector2->raise();
}

bool ZenoHeatMapEditor::eventFilter(QObject* watched, QEvent* event)
{
	if (watched == m_ui->rampSelector2)
	{
		if (event->type() == QEvent::MouseMove)
		{
			QMouseEvent* pMouseEvent = static_cast<QMouseEvent*>(event);
			QPoint newPos = pMouseEvent->pos();
			QRect rcWtf = m_ui->rampSelector2->geometry();
			int X = rcWtf.left();
			int newX = newPos.x();
			newPos.setY(rcWtf.y());
			if (newX > X)
			{
				rcWtf.moveRight(newX);
			}
			else
			{
				rcWtf.moveLeft(newX);
			}
			m_ui->rampSelector2->move(newPos);
			//m_ui->rampSelector2->setGeometry(rcWtf);
			//m_ui->rampSelector2->updateGeometry();
			//m_ui->rampSelector2->update();
		}
	}
	return QWidget::eventFilter(watched, event);
}

void ZenoHeatMapEditor::dragEnterEvent(QDragEnterEvent* event)
{
	if (event->mimeData()->hasFormat("Label"))
		event->acceptProposedAction();
}

void ZenoHeatMapEditor::mousePressEvent(QMouseEvent* event)
{
	QWidget::mousePressEvent(event);
	QWidget* child = childAt(event->pos());
	if (qobject_cast<QLabel*>(child))
		createDrag(event->pos(), child);
}

void ZenoHeatMapEditor::createDrag(const QPoint& pos, QWidget* widget)
{
	if (widget == Q_NULLPTR)
		return;
	QByteArray byteArray(reinterpret_cast<char*>(&widget), sizeof(QWidget*));
	QDrag* drag = new QDrag(this);
	QMimeData* mimeData = new QMimeData;
	mimeData->setData("Label", byteArray);
	drag->setMimeData(mimeData);
	QPoint globalPos = mapToGlobal(pos);
	QPoint p = widget->mapFromGlobal(globalPos);
	drag->setHotSpot(p);
	drag->setPixmap(widget->grab());
	drag->exec(Qt::CopyAction | Qt::MoveAction);
}

void ZenoHeatMapEditor::dropEvent(QDropEvent* event)
{
	QWidget::dropEvent(event);
}