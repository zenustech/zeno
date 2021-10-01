#include "UiDopGraph.h"
#include "UiDopNode.h"
#include "UiDopEditor.h"


void UiDopGraph::select_child(GraphicsWidget *ptr, bool multiselect) {
    GraphicsView::select_child(ptr, multiselect);
    if (editor)
        editor->set_selection(dynamic_cast<UiDopNode *>(ptr));
}
