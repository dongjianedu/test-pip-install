from flask import Flask, jsonify
import threading
import time
import uuid

app = Flask(__name__)
tasks = {}

def long_running_task(uuid):
    # Simulate a long running task
    time.sleep(10)
    print("Task completed")
    tasks[uuid] = "Task completed"

@app.route('/start_task', methods=['POST'])
def start_task():
    task_uuid = str(uuid.uuid4())
    thread = threading.Thread(target=long_running_task, args=(task_uuid,))
    thread.start()
    tasks[task_uuid] = "Task started"
    return jsonify({'task_uuid': task_uuid}), 202

@app.route('/task_status/<task_uuid>', methods=['GET'])
def task_status(task_uuid):
    status = tasks.get(task_uuid, "Task not found")
    return jsonify({'status': status})

if __name__ == '__main__':
    app.run(debug=True)