from setuptools import setup,find_packages

setup(
    name='diffusers_plus',
    version='1.231023.1',
    license='Apache License',
    author="Andrew Zhu",
    author_email='xhinker@hotmail.com',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    url='https://github.com/xhinker/diffusers_plus',
    keywords='diffusers stable-diffusion',
    install_requires=[
        'transformers'
        ,'diffusers'
        , 'opencv-python'
    ],
)