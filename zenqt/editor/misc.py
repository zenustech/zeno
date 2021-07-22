from . import *

CURR_VERSION = 'v1'
MAX_STACK_LENGTH = 100

style = {
    'title_color': '#638e77',
    'socket_connect_color': '#DDDDDD',
    'socket_unconnect_color': '#4a4a4a',
    'title_text_color': '#FFFFFF',
    'title_text_size': 10,
    'button_text_size': 10,
    'socket_text_size': 10,
    'param_text_size': 10,
    'socket_text_color': '#FFFFFF',
    'panel_color': '#282828',
    'blackboard_title_color': '#393939',
    'blackboard_panel_color': '#1B1B1B',
    'line_color': '#B0B0B0',
    'background_color': '#2C2C2C',
    'selected_color': '#EE8844',
    'button_color': '#1e1e1e',
    'button_text_color': '#ffffff',
    'button_selected_color': '#449922',
    'button_selected_text_color': '#333333',
    'output_shift': 1,
    'ramp_width': 10,
    'dummy_socket_width': 5,
    'dummy_socket_height': 40,
    'button_svg_size': 34,
    'button_svg_offset_x': 38,
    'button_svg_offset_y': 38,

    'line_width': 3,
    'ramp_outline_width': 2,
    'node_outline_width': 2,
    'socket_outline_width': 2,
    'node_rounded_radius': 6,
    'socket_radius': 3,
    'socket_offset': 8,
    'node_width': 200,
    'text_height': 23,
    'hori_margin': 9,
    'dummy_socket_offset': 3,
    'dummy_socket_color': '#4D4D4D',
    'top_button_color': '#376557',
    'top_svg_size': 24,
    'top_svg_padding': 5,
}

TEXT_HEIGHT = style['text_height']
HORI_MARGIN = style['hori_margin']
SOCKET_RADIUS = style['socket_radius']
BEZIER_FACTOR = 0.5

def fillRect(painter, rect, color, line_width=None, line_color=None):
    if line_width:
        painter.fillRect(rect, QColor(line_color))

        w = line_width
        r = rect
        content_rect = QRect(r.x() + w, r.y() + w, r.width() - w * 2, r.height() - w * 2)
        painter.fillRect(content_rect, QColor(color))
    else:
        painter.fillRect(rect, QColor(color))

