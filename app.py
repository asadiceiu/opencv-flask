from flask import Flask, render_template, Response
from camera import VideoCap

app = Flask(__name__)
cam = VideoCap(video_path='media/vi_0000_20220725_122016.mp4', refresh_timeout=500)


def camera_frame(camera, video_type: str = "index"):
    """
    Returns a camera frame.
    :param camera:
    :param video_type:
    :return:
    """
    while True:
        if video_type == "index":
            frame = camera.get_frame()
        elif video_type == "opticalflow":
            frame = camera.get_opticalflow()
        elif video_type == "backgroundsubtraction":
            frame = camera.get_background()
        else:
            frame = None
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame +
               b'\r\n\r\n')


@app.route('/')
def index():
    title = "Video"
    return render_template('index.html', title=title)


@app.route('/start_record')
def start_record():
    cam.start_writing  = True
    cam.setup_writer()
    print("start record")
    return {"status": "ok"}


@app.route('/stop_record')
def stop_record():
    cam.start_writing = False
    cam.stop_writer()
    print("stop record")
    return {"status": "ok"}


@app.route('/opticalflow')
def opticalflow():
    title = "Optical Flow"
    return render_template('opticalflow.html', title=title)

# app route for background subtraction
@app.route('/backgroundsubtraction')
def backgroundsubtraction():
    title = "Background Subtraction"
    return render_template('backgroundsubtraction.html', title=title)

@app.route('/video/<string:video_type>')
def video(video_type: str = "index"):
    return Response(camera_frame(cam, video_type), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0')
