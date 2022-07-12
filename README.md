# opencv-flask
A flask server for displaying processed image with OpenCV

 * Creates a flask server
 * serves images from the default webcam
 * uses optical flow dense to track interesting points
 

# putting it into raspberry pi 
## raspberry pi configuration

Set password.
set country using raspi-config
set wifi password and SSID
set interfacing option to enable camera, vnc, ssh support (optional) `sudo raspi-config`

test the camera: `raspistill -o test.jpg`


## install opencv, numpy, flask
`sudo apt-get update`

