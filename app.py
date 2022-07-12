from flask import Flask, render_template, Response
from camera import VideoCap

app = Flask(__name__)
cam = VideoCap(video_path=0, refresh_timeout=500)


def camera_frame(camera, video_type: str = "index"):
    """
    Returns a camera frame.
    :param camera:
    :param video_type:
    :return:
    """
    while True:
        frame = camera.get_frame() if video_type == "index" else camera.get_opticalflow()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame +
               b'\r\n\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/opticalflow')
def opticalflow():
    return render_template('opticalflow.html')


@app.route('/video/<string:video_type>')
def video(video_type: str = "index"):
    return Response(camera_frame(cam, video_type), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0')
