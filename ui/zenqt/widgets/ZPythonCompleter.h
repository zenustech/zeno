#pragma once

// Qt
#include <QPythonCompleter> // Required for inheritance

/**
 * @brief Class, that describes completer with
 * glsl specific types and functions.
 */
class ZPythonCompleter : public QPythonCompleter
{
    Q_OBJECT

public:

    /**
     * @brief Constructor.
     * @param parent Pointer to parent QObject.
     */
    explicit ZPythonCompleter(QObject* parent=nullptr);
};


