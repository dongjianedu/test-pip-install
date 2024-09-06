from lang_sam import LangSAM
from PIL import Image
from PIL import ImageDraw
from PIL import ImageChops
import torch
import uuid
from flask import Flask, jsonify,request,send_file,Response, request, render_template
import json
import subprocess
import os
import numpy as np
import re
import cv2
import dlib
import base64
app = Flask(__name__)
from segment_anything import sam_model_registry, SamPredictor
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(device)
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

sam_predictor = SamPredictor(sam)
#model = LangSAM()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# Load the detector
detector = dlib.get_frontal_face_detector()
request_counter = 0
image_path = "maoshen_blur.jpg"

@app.route("/photo/<imageId>.png",methods=['GET'])
def photo(imageId):
  # 图片上传保存的路径
  print(imageId)
  with open(r'/tmp/{}.png'.format(imageId), 'rb') as f:
    image = f.read()
    print(imageId)
    resp = Response(image, mimetype="image/png")
    return resp
def filter_output(output):
    pattern = r"INFO   \| Job result: \{'output': '/tmp/.*\.png'\}"
    result = re.findall(pattern, output)
    return result

def extract_image_path(output):
    pattern = r"(/tmp/.*\.png)"
    result = re.findall(pattern, output)
    if result:
        return result[0]  # Return the first match
    else:
        return None  # No match found

@app.route('/run', methods=['POST'])
def run():
    global request_counter
    request_counter += 1
    # 获取请求的 payload
    # 获取请求的 payload
    payload = request.get_data(as_text=True)
    print("in run")
    # Check if payload is not empty
    if payload:
        try:
            # Try to parse the payload into a JSON object
            payload = json.loads(payload)
            #payload = {'input': payload}
            with open('./test_input.json', 'w') as f:
                json.dump(payload, f)
            # 打印 payload
        except json.JSONDecodeError as e:
            print("Failed to decode JSON object: ", e)
    else:
        print("No payload received")

    result = subprocess.run(["python", "rp_handler.py"], check=True, capture_output=True, text=True)
    # Print the return value
    print(f"Return code: {result.returncode}")
    print(f"Output: {result.stdout}")
    image_path = extract_image_path(filter_output(result.stdout)[0])
    print(f"image_path: {image_path}")

    print(f"Errors: {result.stderr}")

    response = {
        "id":image_path.split('/')[-1],
        "status":"processing"
    }

    return jsonify(response)


@app.route('/status/<id>', methods=['GET'])
def status(id):
    global request_counter
    request_counter += 1
    # 获取请求的 payload
    # 获取请求的 payload
    print("in status")
    step = 2
    response = {
        "id":"psw5lytbhie3rhlqcvvs4wvswi",
        "model":"timothybrooks/instruct-pix2pix",
        "version":"d-f80397be8373228d825de72f8657f76d",
        "input":{
            "image":"https://user-images.githubusercontent.com/2289/215219780-cb4a0cdb-6fea-46fe-ae22-12d68e5ba79f.jpg",
            "prompt":"make his jacket out of leather"
        },
        "error":None,
        "status":"processing" if request_counter % step else "COMPLETED",
        "created_at":"2024-01-02T05:42:47.883954Z",
        "started_at":"2024-01-02T05:42:47.91775Z",
        "urls":{
            "cancel":"https://api.replicate.com/v1/predictions/psw5lytbhie3rhlqcvvs4wvswi/cancel",
            "get":"https://api.replicate.com/v1/predictions/psw5lytbhie3rhlqcvvs4wvswi"
        }
    }

    if request_counter % step == 0:
        response["output"] = [f"http://127.0.0.1:8080/photo/{id}","https://f005.backblazeb2.com/file/demo-image/maoshen_after.png"]
    print(response)
    return jsonify(response)



def create_face_mask(image_path):
    # Load image
    img = cv2.imread(image_path)

    # Convert image to grayscale
    gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

    # Use detector to find faces in the image
    faces = detector(gray)

    # This will store the final mask
    mask = np.zeros_like(gray)

    # Variables to store the highest x and y coordinates where the mask value is not zero
    highest_x = 0
    highest_y = 0

    for face in faces:
        # Use predictor to find landmark points
        landmarks = predictor(image=gray, box=face)

        # Create an array of landmark points
        landmark_points = []
        nose_points = []
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmark_points.append((x, y))

            # Update the highest x and y coordinates where the mask value is not zero
            if n >= 27 and n <= 35:  # these are the points for the nose
                nose_points.append((x, y))

        # Convert landmark points to a numpy array
        points = np.array(landmark_points, np.int32)

        # Create a convex hull around the points
        hull = cv2.convexHull(points)

        # Draw the convex hull on the mask
        cv2.drawContours(mask, [hull], 0, (255, 255, 255), -1)

        # Calculate the center of the nose
        if nose_points:
            highest_x, highest_y = np.mean(nose_points, axis=0)

    return mask, int(highest_x), int(highest_y)

def merge_images(mask_image_with_dot, face_masks_image, highest_y):
    # Ensure mask_image_with_dot and face_masks_image have the same size
    assert mask_image_with_dot.size == face_masks_image.size

    # Convert the images to numpy arrays
    mask_array = np.array(mask_image_with_dot)
    face_masks_array = np.array(face_masks_image)


    # Create a new array with the same shape
    merged_array = np.zeros_like(mask_array)

    # Fill the new array with values from mask_array or face_masks_array
    for y in range(merged_array.shape[0]):
        if y  > highest_y:
            merged_array[y] = mask_array[y]
        else:
            merged_array[y] = face_masks_array[y]

    # Convert the merged array to a PIL Image
    merged_image = Image.fromarray(merged_array)

    # Save the image
    return merged_image

def get_output_layers(net):
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

def get_max_box(image_path, config_path, weights_path, classes_path):
    image = cv2.imread(image_path)

    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    classes = None

    with open(classes_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    net = cv2.dnn.readNet(weights_path, config_path)

    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    max_area = 0
    max_box = None

    for box in boxes:
        x, y, w, h = box
        area = w * h
        if area > max_area:
            max_area = area
            max_box = box

    x1 = max_box[0]
    y1 = max_box[1]
    x2 = max_box[0] + max_box[2]
    y2 = max_box[1] + max_box[3]

    return x1, y1, x2, y2


def get_largest_face_box(image_path):
    # Load the detector
    detector = dlib.get_frontal_face_detector()

    # Load image
    img = cv2.imread(image_path)

    # Convert image to grayscale
    gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

    # Use detector to find faces in the image
    faces = detector(gray)

    print("faces detected: "+str(len(faces)))

    largest_box = None
    largest_area = 0
    # Loop through each face
    for face in faces:
        # Get the coordinates of rectangle corners
        x1 = face.left()  # left point
        y1 = face.top()  # top point
        x2 = face.right()  # right point
        y2 = face.bottom()  # bottom point

        # Calculate the width and height of the face
        width = x2 - x1
        height = y2 - y1

        # Calculate the new coordinates for a box that is 1.5 times the width and 2 times the height of the face
        new_x1 = max(0, x1 - int(width * 0.25))
        new_y1 = max(0, y1 - int(height * 0.5))
        new_x2 = min(img.shape[1], x2 + int(width * 0.25))
        new_y2 = min(img.shape[0], y2 + int(height * 0.5))

        # Calculate the area of the box
        area = (new_x2 - new_x1) * (new_y2 - new_y1)

        # If this box has the largest area so far, save it
        if area > largest_area:
            largest_area = area
            largest_box = (new_x1, new_y1, new_x2, new_y2)

    return largest_box

def decode_and_save_base64(base64_str, save_dir='/tmp'):
    # Generate a unique filename using uuid
    filename = str(uuid.uuid4()) + ".png"
    save_path = os.path.join(save_dir, filename)

    with open(save_path, "wb") as file:
        file.write(base64.b64decode(base64_str))

    return save_path  # Return the path of the saved file

def encode_file_to_base64(path):
    with open(path, 'rb') as file:
        return base64.b64encode(file.read()).decode('utf-8')

@app.route('/process', methods=['POST', 'GET'])
def process_image():
    print("in process")
    image_path = request.json['image_path']
    base64_image = request.json['base64']
    if(base64_image):
        image_path = decode_and_save_base64(base64_image)
        print("use base 64"+image_path)
    # Convert the base64 string to a byte array

    x1, y1, x2, y2 = get_largest_face_box(image_path)
    print(x1, y1, x2, y2)
    input_head_box = np.array([x1, y1, x2, y2])

    x1_, y1_, x2_, y2_ = get_max_box(image_path, "yolov3.cfg", "yolov3.weights", "yolov3.txt")
    input_box = np.array([x1_, y1_, x2_, y2_])
    print(x1_, y1_, x2_, y2_)
    mask, highest_x, highest_y = create_face_mask(image_path)

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    sam_predictor.set_image(image)

    input_point = np.array([[highest_x, highest_y]])
    input_label = np.array([1])

    face_sam_masks, _, _ = sam_predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        box=None,
        multimask_output=False,
    )

    face_sam_masks_2d = face_sam_masks.astype(float).squeeze(0)
    face_sam_masks_image = Image.fromarray((face_sam_masks_2d * 255).astype(np.uint8))
    face_sam_masks_image.save("/tmp/face_sam_masks.png")

    body_sam_masks, _, _ = sam_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False,
    )

    body_sam_masks_2d = body_sam_masks.astype(float).squeeze(0)
    body_sam_masks_image = Image.fromarray((body_sam_masks_2d * 255).astype(np.uint8))
    body_sam_masks_image.save("/tmp/body_sam_masks.png")
    random_uuid = uuid.uuid4()
    random_file_name = str(random_uuid) + '.png'
    body_sam_masks_path = f"/tmp/{random_file_name}"
    body_sam_masks_image.save(body_sam_masks_path)
    body_sam_masks_base64 = encode_file_to_base64(body_sam_masks_path)


    head_sam_masks, _, _ = sam_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_head_box[None, :],
        multimask_output=False,
    )

    head_sam_masks_2d = head_sam_masks.astype(float).squeeze(0)
    head_sam_masks_image = Image.fromarray((head_sam_masks_2d * 255).astype(np.uint8))
    head_sam_masks_image.save("/tmp/head_sam_masks.png")
    print(highest_x, highest_y)
    mask_tensor = torch.from_numpy(mask)
    # # Convert tensor to numpy array
    mask_array = mask_tensor.numpy()

    # Remove extra dimensions
    mask_array = np.squeeze(mask_array)

    # Convert numpy array to PIL Image
    mask_image = Image.fromarray(mask_array)
    print(mask_image.size)
    mask_image.save("/tmp/mask_image.png")




    merged_image = merge_images(mask_image, face_sam_masks_image, highest_y)

    image_pil = Image.open(image_path).convert("RGB")

    #body_masks, body_boxes, body_phrases, body_logits = model.predict(image_pil, 'woman')




    merged_image_2 = ImageChops.add(head_sam_masks_image, merged_image)

    # Invert the merged image
    inverted_image = ImageChops.invert(merged_image_2)

    # Save the inverted image
    inverted_image.save("/tmp/inverted_image.png")


    result_mask_image = ImageChops.darker(inverted_image, body_sam_masks_image)
    random_uuid = uuid.uuid4()
    random_file_name = str(random_uuid) + '.png'
    path = f"/tmp/{random_file_name}"
    result_mask_image.save(path)
    mask_base64 = encode_file_to_base64(path)

    image_array = np.array(image_pil)

    mask_array = np.array(face_sam_masks_image)
    mask_array = np.repeat(np.expand_dims(mask_array, axis=-1), 3, axis=-1)
    body_cropped_image_array = np.where(mask_array == 255, image_array, 0)
    face_cropped_image = Image.fromarray(body_cropped_image_array)
    #face_cropped_image.save("./result/face_cropped_image.png")

    cropped_image_gray = face_cropped_image.convert("L")

    # Convert the grayscale image to a numpy array
    cropped_image_array = np.array(cropped_image_gray)

    # Calculate the average brightness
    non_zero_values = cropped_image_array[cropped_image_array.nonzero()]
    average_brightness = non_zero_values.mean()
    face_average_brightness = int(average_brightness)
    print("face Average Brightness:", average_brightness)


    mask_array = np.array(body_sam_masks_image)
    mask_array = np.repeat(np.expand_dims(mask_array, axis=-1), 3, axis=-1)
    body_cropped_image_array = np.where(mask_array == 255, image_array, face_average_brightness)
    body_cropped_image = Image.fromarray(body_cropped_image_array)
    random_uuid = uuid.uuid4()
    random_file_name = str(random_uuid) + '.png'
    body_cropped_image_path = f"/tmp/{random_file_name}"
    body_cropped_image.save(body_cropped_image_path)
    white_image_base64 = encode_file_to_base64(body_cropped_image_path)

    mask_array = np.array(result_mask_image)
    mask_array = np.repeat(np.expand_dims(mask_array, axis=-1), 3, axis=-1)

    # Use the mask to crop the image
    cropped_image_array = np.where(mask_array == 255, image_array, 0)

    # Convert the cropped image array back to PIL Image
    cropped_image = Image.fromarray(cropped_image_array)

    # Save the cropped image
    cropped_image.save("/tmp/cropped_image.png")

    # Convert the image to grayscale
    cropped_image_gray = cropped_image.convert("L")

    # Convert the grayscale image to a numpy array
    cropped_image_array = np.array(cropped_image_gray)

    # Calculate the average brightness
    average_brightness = cropped_image_array.mean()

    print("Average Brightness:", average_brightness)
    response = {
        'path': path,
        'mask_base64': mask_base64,
        'body_cropped_image_path':body_cropped_image_path,
        'white_image_base64':white_image_base64,
        'face_average_brightness': face_average_brightness,
        'body_sam_masks_path': body_sam_masks_path,
        'body_sam_masks_base64': body_sam_masks_base64,
        'average_brightness': average_brightness,
        'mask_status':True
    }

    return response, 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080,debug=True)