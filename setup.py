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
        "transformers",
        "torch",
        "click",
        "tabulate",
        "sklearn",
        "jieba"
    ],
    package_dir={'textclf': 'textclf'},
    python_requires='>=3',
    entry_points={
        'console_scripts': ['textclf=textclf.main:main'],
    }
)
