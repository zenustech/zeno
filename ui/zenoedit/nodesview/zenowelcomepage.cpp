#include "ui_welcomepage.h"
#include "zenowelcomepage.h"


ZenoWelcomePage::ZenoWelcomePage(QWidget* parent)
	: QWidget(parent)
{
	m_ui = new Ui::WelcomePage;
	m_ui->setupUi(this);

	initSignals();
}

void ZenoWelcomePage::initSignals()
{
	connect(m_ui->btnNew, SIGNAL(clicked()), this, SIGNAL(newRequest()));
	connect(m_ui->btnOpen, SIGNAL(clicked()), this, SIGNAL(openRequest()));
}