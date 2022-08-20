# YOLO_v3_Marker_Detector

This project is used to detect markers based on live video or any photo given by the user.

First, marker photos were taken and Labelimg was used to bound the markers in boxes in turn generating their coordinates.

The coordinates were trained in Google Colab using YOLO v3.

Configuration file of the training as well as weights were downloaded and used locally.

Finally, marker was detected with codes from test_on_video.



### Steps to run
Master branch includes all codes file.

### Clone this [Repo](https://github.com/Anushil007/YOLO_v3_Marker_Detector)
### Create Virtual Enviroment and activate
python -m venv YOLO_env

### Install requirements.txt
pip install -r requirements.txt

### To detect marker on live vioeo run
test_on_video.py

## To detect marker on  Image run and give image path
test_on_photo.py
