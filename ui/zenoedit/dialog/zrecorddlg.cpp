#include "zrecorddlg.h"
#include "ui_zrecorddlg.h"
#include <QFileDialog>
#include "zenoapplication.h"
#include "zenomainwindow.h"


ZRecordVideoDlg::ZRecordVideoDlg(int frameStart, int frameEnd, QWidget* parent)
	: QDialog(parent)
{
	m_ui = new Ui::RecordVideoDlg;
	m_ui->setupUi(this);

	m_ui->frameStart->setValidator(new QIntValidator(frameStart, frameEnd));
	m_ui->frameStart->setText(QString::number(frameStart));
	m_ui->frameEnd->setValidator(new QIntValidator(frameStart, frameEnd));
	m_ui->frameEnd->setText(QString::number(frameEnd));
	m_ui->fps->setValidator(new QIntValidator);
	m_ui->fps->setText("24");
	m_ui->bitrate->setValidator(new QIntValidator);
	m_ui->bitrate->setText("20000");
	m_ui->lineWidth->setValidator(new QIntValidator);
	m_ui->lineWidth->setText("1280");
	m_ui->lineHeight->setValidator(new QIntValidator);
	m_ui->lineHeight->setText("720");
    m_ui->msaaSamplerNumber->setValidator(new QIntValidator);
    m_ui->msaaSamplerNumber->setText("0");
    m_ui->optixSamplerNumber->setValidator(new QIntValidator);
    m_ui->optixSamplerNumber->setText("1");

	m_ui->cbPresets->addItems({"540P", "720P", "1080P", "2K", "4K"});
	m_ui->cbPresets->setCurrentIndex(1);

    connect(m_ui->cbPresets, &QComboBox::currentTextChanged, this, [=](auto res) {
        auto v = std::map<QString, std::tuple<int, int>> {
                {"540P", {960, 540}},
                {"720P", {1280, 720}},
                {"1080P", {1920, 1080}},
                {"2K", {2560, 1440}},
                {"4K", {3840, 2160}},
        }.at(res);
        m_ui->lineWidth->setText(QString::number(std::get<0>(v)));
        m_ui->lineHeight->setText(QString::number(std::get<1>(v)));
    });

	connect(m_ui->btnOpen, &QPushButton::clicked, this, [=]() {
		DlgInEventLoopScope;
		QString path = QFileDialog::getExistingDirectory(nullptr, tr("File to Save"), "");
		if (path.isEmpty())
			return;
		m_ui->linePath->setText(path);
	});

	connect(m_ui->btnGroup, SIGNAL(accepted()), this, SLOT(accept()));
	connect(m_ui->btnGroup, SIGNAL(rejected()), this, SLOT(reject()));
}

bool ZRecordVideoDlg::getInfo(int& frameStart, int& frameEnd, int& fps, int& bitrate, QString& presets, int& width, int& height, QString& path, QString& fn, int &numOptix, int &numMSAA)
{
	frameStart = m_ui->frameStart->text().toInt();
	frameEnd = m_ui->frameEnd->text().toInt();
	fps = m_ui->fps->text().toInt();
	bitrate = m_ui->bitrate->text().toInt();
    numMSAA = m_ui->msaaSamplerNumber->text().toInt();
    numOptix = m_ui->optixSamplerNumber->text().toInt();
	presets = m_ui->cbPresets->currentText();
	width = m_ui->lineWidth->text().toInt();
	height = m_ui->lineHeight->text().toInt();
	path = m_ui->linePath->text();
	if (path.isEmpty())
	{
		QTemporaryDir dir;
		dir.setAutoRemove(false);
		path = dir.path();
	}
	fn = m_ui->lineName->text();
	if (fn.isEmpty())
	{
		fn = "capture";
		const QString& suffix = ".mp4";
		int idx = 1;
		while (QFileInfo(path + "/" + fn + suffix).exists())
		{
			fn = "capture_" + QString::number(idx);
			idx++;
		}
		fn += suffix;
	}
	return true;
}