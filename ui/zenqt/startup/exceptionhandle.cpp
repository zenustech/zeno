#include "exceptionhandle.h"

#if defined(Q_OS_WIN)

#include <windows.h>
#include <Dbghelp.h>

QTemporaryDir sTempDir;

LONG WINAPI MyUnhandledExceptionFilter(struct _EXCEPTION_POINTERS* ExceptionInfo)
{
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

void registerExceptionFilter()
{
    SetUnhandledExceptionFilter(MyUnhandledExceptionFilter);
    sTempDir.setAutoRemove(false);
}

#endif