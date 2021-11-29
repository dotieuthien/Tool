LabelComponent
==========================================

Requirements
------------

- Ubuntu
- Python3
- [PyQt5](http://www.riverbankcomputing.co.uk/software/pyqt/intro)
- labelme3.yml


Installation
------------

You need install [Anaconda](https://www.continuum.io/downloads), then run below:

```
# Python3
cd $root_dir_to_LabelComponent
conda create --name=labelme3 python=3.6
conda activate labelme3
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
 

