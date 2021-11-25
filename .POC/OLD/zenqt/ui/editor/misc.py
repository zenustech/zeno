from . import *

CURR_VERSION = 'v1'
MAX_STACK_LENGTH = 100

class BackgroundStyle(object):
    NONE = 0
    DOT = 1

style = {
    'title_color': '#638e77',
    'socket_connect_color': '#638e77',
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
    'background_color': '#263238',
    'selected_color': '#EE8844',
    'button_color': '#1e1e1e',
    'button_text_color': '#ffffff',
    'button_selected_color': '#449922',
    'button_selected_text_color': '#333333',
    'output_shift': 1,
    'ramp_width': 10,

    'line_width': 3,
    'ramp_outline_width': 2,
    'node_outline_width': 2,
    'socket_outline_width': 2,
    'node_rounded_radius': 6,
    'socket_radius': 8,
    'node_width': 200,
    'text_height': 23,
    'hori_margin': 9,
    'dummy_socket_offset': 15,

    'background_style': BackgroundStyle.DOT,
}

TEXT_HEIGHT = style['text_height']
HORI_MARGIN = style['hori_margin']
SOCKET_RADIUS = style['socket_radius']
BEZIER_FACTOR = 0.5

