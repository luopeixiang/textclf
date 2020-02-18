from setuptools import setup, find_packages


# packages=find_packages(exclude=['test']),
setup(
    name='textclf',
    author="luopeixiang",
    author_email="luopeixiang@qq.com",
    version="0.1.0",
    description='文本分类工具集',
    packages=["textclf", "textclf.data"],
    install_requires=[
        "transformers==2.4.1",
        "torch>=1.1.0",
        "Click==7.0.0",
        "tabulate==0.8.6",
        "scikit-learn==0.22.1",
        "jieba"
    ],
    package_dir={'textclf': 'textclf'},
    python_requires='>=3.6',
    entry_points={
        'console_scripts': ['textclf=textclf.main:main'],
    }
)
