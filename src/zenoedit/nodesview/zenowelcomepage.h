#ifndef __ZENO_WELCOME_PAGE_H__
#define __ZENO_WELCOME_PAGE_H__

namespace Ui
{
	class WelcomePage;
}

class ZenoWelcomePage : public QWidget
{
public:
	ZenoWelcomePage(QWidget* parent = nullptr);

private:
	Ui::WelcomePage* m_ui;
};


#endif