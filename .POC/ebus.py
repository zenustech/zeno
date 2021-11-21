class FrameChangedEvent:
    def __init__(self, frameid):
        self.frameid = frameid


class EventBus:
    def __init__(self):
        self.subscribers = {}

    def _subscribe(self, event_type, user):
        self.subscribers.setdefault(event_type, set()).add(user)

    def _unsubscribe(self, event_type, user):
        self.subscribers.get(event_type, set()).remove(user)

    def send(self, event):
        event_type = type(event).__name__
        for user in self.subscribers.get(event_type, set()):
            getattr(user, 'on{}'.format(event_type))(event)

g_bus = EventBus()


class EventUser:
    def __init__(self):
        self.__subscribed = []

    def subscribe(self, event_type):
        g_bus._subscribe(event_type, self)
        self.__subscribed.append(event_type)

    def __deinit__(self):
        for event_type in self.__subscribed:
            g_bus._unsubscribe(event_type, self)
        del self.__subscribed


class TimelineWidget(EventUser):
    def __init__(self):
        super().__init__()
        self.subscribe('FrameChangedEvent')

    def onFrameChangedEvent(self, event):
        print('TimelineWidget got new frameid:', event.frameid)


class SliderWidget(EventUser):
    def __init__(self):
        super().__init__()

    def mouseDragTo(self, frameid):
        g_bus.send(FrameChangedEvent(frameid))


class DialogWidget(EventUser):
    def __init__(self):
        super().__init__()

    def userInputIs(self, frameid):
        g_bus.send(FrameChangedEvent(frameid))


timeline = TimelineWidget()
slider = SliderWidget()
dialog = DialogWidget()

dialog.userInputIs(32)
slider.mouseDragTo(42)
