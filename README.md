## 🚀&nbsp; Installation
1. Clone the repo
```
$ git clone https://github.com/RitwikSingh12/face_recog_and_mask_detection.git
```

2. Change your directory to the cloned repo 
```
$ cd face_recog_and_mask_detection
```

3. Create a Python virtual environment named 'test' and activate it
```
$ virtualenv env
```
```
$ source test/bin/activate
```

4. Now, run the following command in your Terminal/Command Prompt to install the libraries required
```
$ pip install -r requirements.txt
```

## :bulb: Working

1. Open terminal. Go into the cloned project directory and type the following command:
```
$ python train_mask_detector.py --dataset dataset
```

2. To detect face masks in an image type the following command: 
```
$ python detect_mask_image.py --image images/pic1.jpeg
```

3. To detect face masks in real-time video streams type the following command:
```
$ python detect_mask_video.py 
```
