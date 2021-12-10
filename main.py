from flask import Flask, render_template, Response
app=Flask(__name__)
import face_detection as fd
import face_recog as fr

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/face_d')
def video_feed():
    return Response(fd.main(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/img_fr')
def video_fr():
    return Response(fr.main(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__=='__main__':
    app.run(debug=True)