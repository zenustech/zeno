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
            'image',
    ]

    def apply(self):
        path = self.get_param('path')
        image = cv2.imread(path)
        self.set_output('image', image)


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
    ]
    z_inputs = [
            'image',
    ]
    z_outputs = [
    ]

    def apply(self):
        title = self.get_param('title')
        image = self.get_input('image')
        cv2.imshow(title, image)


@zen.defNodeClass
class CV_waitKey(zen.INode):
    z_categories = 'opencv'
    z_params = [
            ('delay_ms', 'int', '0 0'),
    ]
    z_inputs = [
    ]
    z_outputs = [
    ]

    def apply(self):
        delay = self.get_param('delay_ms')
        cv2.waitKey(delay)



@zen.defNodeClass
class CV_cvtColorBGR2GRAY(zen.INode):
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
        path = self.get_param('path')
        image = self.get_input('image')
        result = cv2.cvtColor(image, cv2.BGR2GRAY)
        self.set_output('result', result)


if __name__ == '__main__':
    print(zen.dumpDescriptors())
