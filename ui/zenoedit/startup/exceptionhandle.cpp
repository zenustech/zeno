#include "exceptionhandle.h"

#include "cache/zcachemgr.h"
#include "zenoapplication.h"

#if defined(Q_OS_WIN)

#include <windows.h>
#include <Dbghelp.h>

QTemporaryDir sTempDir;

LONG WINAPI MyUnhandledExceptionFilter(struct _EXCEPTION_POINTERS* ExceptionInfo)
{
    //清理缓存
	std::shared_ptr<ZCacheMgr> mgr = zenoApp->cacheMgr();
	mgr->procExitCleanUp();

    QMessageBox::information(nullptr, "Crash", "Zeno has crashed, recording dump info right now...");

    QDateTime dateTime = QDateTime::currentDateTime();
    QString timestamp = dateTime.toString("yyyy-MM-dd-hh-mm-ss");
    QString fileName = QString("crash_%1.dmp").arg(timestamp);
    QString filePath = sTempDir.filePath(fileName);

    HANDLE hFile = CreateFile(filePath.toStdString().c_str(),
                              GENERIC_READ | GENERIC_WRITE, FILE_SHARE_WRITE, NULL,
                              CREATE_ALWAYS,
                              FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE)
        return EXCEPTION_EXECUTE_HANDLER;

    MINIDUMP_EXCEPTION_INFORMATION mdei;
    mdei.ThreadId = GetCurrentThreadId();
    mdei.ExceptionPointers = ExceptionInfo;
    mdei.ClientPointers = NULL;

    MINIDUMP_CALLBACK_INFORMATION mci;
    mci.CallbackRoutine = NULL;
    mci.CallbackParam = 0;

    MiniDumpWriteDump(GetCurrentProcess(), GetCurrentProcessId(), hFile, MiniDumpNormal, &mdei, NULL, &mci);
    CloseHandle(hFile);

    QDesktopServices::openUrl(QUrl::fromLocalFile(sTempDir.path()));
    return EXCEPTION_EXECUTE_HANDLER;
}

// 控制台事件处理：处理关闭命令窗口等强制退出事件
BOOL WINAPI ConsoleHandler(DWORD ctrlType) {
	switch (ctrlType) {
	case CTRL_CLOSE_EVENT:  // 关闭命令窗口
	case CTRL_C_EVENT:      // Ctrl+C强制终止
    case CTRL_BREAK_EVENT:  // Ctrl+Break强制终止
    case CTRL_SHUTDOWN_EVENT:
    {
		std::shared_ptr<ZCacheMgr> mgr = zenoApp->cacheMgr();
		mgr->procExitCleanUp();
		// 给清理操作留一定时间，然后允许进程退出
		//Sleep(300); // 调整等待时间（根据清理逻辑耗时）
		return TRUE;
    }
	default:
		return FALSE;
	}
}

void registerExceptionFilter()
{
    SetUnhandledExceptionFilter(MyUnhandledExceptionFilter);
    SetConsoleCtrlHandler(ConsoleHandler, TRUE);
    sTempDir.setAutoRemove(false);
}

#endif