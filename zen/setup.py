import setuptools
import glob

data_files = ['zen/libzenpy.so'] + glob.glob('zen/include/zen/*')

setuptools.setup(name='zen',
                 packages=['zen'],
                 version='0.0.1',
                 description='Zensim node system',
                 author='archibate',
                 author_email='1931127624@qq.com',
                 url='https://github.com/littlemine/Mn',
                 install_requires=[
                     'pybind11>=2.5.0',
                 ],
                 data_files=data_files,
                 keywords=['graphics', 'simulation'],
                 include_package_data=True,
                 classifiers=[
                     'Topic :: Multimedia :: Graphics',
                     'Topic :: Games/Entertainment :: Simulation',
                     'Intended Audience :: Science/Research',
                     'Intended Audience :: Developers',
                     'Programming Language :: Python :: 3.9',
                 ],
                 has_ext_modules=lambda: True)
