async_mode = None

import time
from flask import Flask, render_template
import socketio

sio = socketio.Server(logger=True, async_mode=async_mode)
app = Flask(__name__)
app.wsgi_app = socketio.Middleware(sio, app.wsgi_app)
app.config['SECRET_KEY'] = 'secret!'
thread = None


def background_thread():
    sio.emit('message', {'data': 'Program started'})
    # main(display_data)
    sio.sleep(2)
    send_message('Another way to send data')
    init_plots([0,1,2,3,4,5,6,7,8,9,10,11]);

def send_message(message):
    sio.emit('message', {'data': message})

def send_data(data):
    sio.emit('message', {'data': 'TBI'})

def init_plots(titles):
    sio.emit('init plots', {'data': titles})

@app.route('/')
def index():
    return render_template('index.html')

@sio.on('connected', namespace='')
def test_message(sid, message):
    print('Client connected')

@sio.on('connect', namespace='')
def test_connect(sid, environ):
    sio.emit('message', {'data': 'Server connected, waiting to start', 'count': 0})

@sio.on('disconnect request', namespace='/test')
def disconnect_request(sid):
    sio.disconnect(sid, namespace='/test')

@sio.on('disconnect', namespace='')
def test_disconnect(sid):
    print('Client disconnected')

@sio.on('initialize', namespace='')
def initialize(sid):
    global thread
    thread = sio.start_background_task(background_thread)
    # if thread is None:
        # thread = sio.start_background_task(background_thread)


if __name__ == '__main__':
    if sio.async_mode == 'threading':
        # deploy with Werkzeug
        app.run(threaded=True)
    elif sio.async_mode == 'eventlet':
        # deploy with eventlet
        import eventlet
        import eventlet.wsgi
        eventlet.wsgi.server(eventlet.listen(('', 3000)), app)
    elif sio.async_mode == 'gevent':
        # deploy with gevent
        from gevent import pywsgi
        try:
            from geventwebsocket.handler import WebSocketHandler
            websocket = True
        except ImportError:
            websocket = False
        if websocket:
            pywsgi.WSGIServer(('', 3000), app,
                              handler_class=WebSocketHandler).serve_forever()
        else:
            pywsgi.WSGIServer(('', 3000), app).serve_forever()
    elif sio.async_mode == 'gevent_uwsgi':
        print('Start the application through the uwsgi server. Example:')
        print('uwsgi --http :3000 --gevent 1000 --http-websockets --master '
              '--wsgi-file app.py --callable app')
    else:
        print('Unknown async_mode: ' + sio.async_mode)
