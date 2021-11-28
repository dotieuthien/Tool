labelme: Image Annotation Tool with Python
==========================================

Requirements
------------

- Ubuntu
- Python2
- [PyQt5](http://www.riverbankcomputing.co.uk/software/pyqt/intro)


Installation
------------

You need install [Anaconda](https://www.continuum.io/downloads), then run below:

```
# Python2
cd $root_dir_of_labelme
conda create --name=labelme python=2.7
source activate labelme
conda install pyqt
pip install labelme
```
Re-Install labelme
```
# Install
$ pip uninstall labelme
$ python setup.py install
```

Usage
-----
Data input has the format:
```
# Add color folder because this is the format of Geektoys data, the extentions = ['png', 'jpg', 'jpeg', 'tga']
your_data_dir/color/your_images
```

```
# Run
$ PYTHONPATH=.:${PYTHONPATH}
$ export PYTHONPATH
$ python labelme/app.py
```
 

