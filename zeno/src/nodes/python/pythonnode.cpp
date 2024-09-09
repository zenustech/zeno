#ifdef ZENO_WITH_PYTHON
#include <Python.h>
#include <zeno/zeno.h>
#include <zeno/utils/log.h>


#if 0
void onExecuteClicked()
{
    std::string stdOutErr =
        "import sys\n\
class CatchOutErr:\n\
    def __init__(self):\n\
        self.value = ''\n\
    def write(self, txt):\n\
        self.value += txt\n\
    def flush(self):\n\
        pass\n\
catchOutErr = CatchOutErr()\n\
sys.stdout = catchOutErr\n\
sys.stderr = catchOutErr\n\
"; //this is python code to redirect stdouts/stderr

    Py_Initialize();
    PyObject* pModule = PyImport_AddModule("__main__"); //create main module
    PyRun_SimpleString(stdOutErr.c_str()); //invoke code to redirect

    ParamsModel* paramsM = QVariantPtr<ParamsModel>::asPtr(index().data(ROLE_PARAMS));
    ZASSERT_EXIT(paramsM);
    QModelIndex scriptIdx = paramsM->paramIdx("script", true);
    ZASSERT_EXIT(scriptIdx.isValid());
    QString script = scriptIdx.data(ROLE_PARAM_VALUE).toString();

    //Py_Initialize();
    if (PyRun_SimpleString(script.toUtf8()) < 0) {
        zeno::log_warn("Python Script run failed");
    }
    PyObject* catcher = PyObject_GetAttrString(pModule, "catchOutErr"); //get our catchOutErr created above
    PyObject* output = PyObject_GetAttrString(catcher, "value"); //get the stdout and stderr from our catchOutErr object
    if (output != Py_None)
    {
        QString str = QString::fromStdString(_PyUnicode_AsString(output));
        QStringList lst = str.split('\n');
        for (const auto& line : lst)
        {
            if (!line.isEmpty())
            {
                if (zenoApp->isUIApplication())
                    ZWidgetErrStream::appendFormatMsg(line.toStdString());
            }
        }
    }

    zeno::log_warn("The option 'ZENO_WITH_PYTHON3' should be ON");

}
#endif

#endif