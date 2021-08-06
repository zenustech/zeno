import cv2
import zen.py as zen


@zen.defNodeClass
class CV_imread(zen.INode):
    z_categories = 'opencv'
    z_params = [
            ('path', 'str', ''),
    ]
    z_inputs = [
    ]
    z_outputs = [
            'result',
    ]

    def apply(self):
        path = self.get_param('path')
        result = cv2.imread(path)
        assert result is not None, path
        self.set_output('result', result)


@zen.defNodeClass
class CV_imwrite(zen.INode):
    z_categories = 'opencv'
    z_params = [
            ('path', 'str', ''),
    ]
    z_inputs = [
            'image',
    ]
    z_outputs = [
    ]

    def apply(self):
        path = self.get_param('path')
        image = self.get_input('image')
        cv2.imwrite(path, image)


@zen.defNodeClass
class CV_imshow(zen.INode):
    z_categories = 'opencv'
    z_params = [
            ('title', 'str', 'imshow'),
            ('delay_ms', 'int', '0 -1'),
    ]
    z_inputs = [
            'image',
    ]
    z_outputs = [
    ]

    def apply(self):
        title = self.get_param('title')
        image = self.get_input('image')
        delay = self.get_param('delay_ms')
        cv2.imshow(title, image)
        if delay >= 0:
            cv2.waitKey(delay)


@zen.defNodeClass
class CV_cvtBGR2GRAY(zen.INode):
    z_categories = 'opencv'
    z_params = [
            ('path', 'str', ''),
    ]
    z_inputs = [
            'image',
    ]
    z_outputs = [
            'result',
    ]

    def apply(self):
        image = self.get_input('image')
        result = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.set_output('result', result)


@zen.defNodeClass
class CV_invertColor(zen.INode):
    z_categories = 'opencv'
    z_params = [
            ('path', 'str', ''),
    ]
    z_inputs = [
            'image',
    ]
    z_outputs = [
            'result',
    ]

    def apply(self):
        image = self.get_input('image')
        result = 255 - image
        self.set_output('result', result)


if __name__ == '__main__':
    zen.addNode("CV_imread", "read")
    zen.setNodeParam("read", "path", "/home/bate/Pictures/face.png")
    zen.applyNode("read")
    zen.addNode("CV_cvtBGR2GRAY", "proc")
    #zen.addNode("CV_invertColor", "proc")
    zen.setNodeInput("proc", "image", "read::result")
    zen.applyNode("proc")
    zen.addNode("CV_imshow", "show")
    zen.setNodeParam("show", "title", "imshow")
    zen.setNodeParam("show", "delay_ms", 0)
    zen.setNodeInput("show", "image", "proc::result")
    zen.applyNode("show")
    print(zen.dumpDescriptors())
