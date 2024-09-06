import time
import base64
import runpod
import requests
import json
import os
from requests.adapters import HTTPAdapter, Retry
from datetime import datetime
import warnings
from PIL import Image, ImageFilter
import base64
import torch
from io import BytesIO
from PIL import ImageChops
import uuid
import threading
import signal
import sys
import numpy as np
from runpod.serverless.utils.rp_upload import upload_image
LOCAL_URL = "http://pi.mytunnel.top:3000/sdapi/v1"
SAM_URL = "http://127.0.0.1:8080/process"
no_face_url = "https://f005.backblazeb2.com/file/demo-image/no_face.png"
out_dir = 'api_out'
out_dir_t2i = os.path.join(out_dir, 'txt2img')
automatic_session = requests.Session()
retries = Retry(total=10, backoff_factor=0.1, status_forcelist=[502, 503, 504])
automatic_session.mount('http://', HTTPAdapter(max_retries=retries))


# ---------------------------------------------------------------------------- #
#                              Automatic Functions                             #
# ---------------------------------------------------------------------------- #
def wait_for_service(url):
    '''
    Check if the service is ready to receive requests.
    '''
    while True:
        try:
            requests.get(url, timeout=120)
            return
        except requests.exceptions.RequestException:
            print("Service not ready yet. Retrying...")
        except Exception as err:
            print("Error: ", err)

        time.sleep(0.2)


def apply_glass_effect(image_path, radius):
    # 打开图像
    img = Image.open(image_path)
    # 应用毛玻璃效果
    img_blurred = img.filter(ImageFilter.GaussianBlur(radius))
    return img_blurred

def run_inference(inference_request):
    '''
    Run inference on a request.
    '''
    response = automatic_session.post(url=f'{LOCAL_URL}/img2img',
                                      json=inference_request, timeout=600)
    return response.json()

def timestamp():
    return datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")




def decode_and_save_base64(base64_str, save_dir='/tmp'):
    # Generate a unique filename using uuid
    filename = str(uuid.uuid4()) + ".png"
    save_path = os.path.join(save_dir, filename)

    with open(save_path, "wb") as file:
        file.write(base64.b64decode(base64_str))

    return save_path  # Return the path of the saved file



def tensor_to_base64(tensor):
    '''
    Convert a tensor to base64 string.
    '''
    # Convert the tensor to a PIL image
    mask_image = Image.fromarray((tensor * 255).type(torch.uint8).numpy())

    # Save the PIL image to a BytesIO object
    byte_arr = BytesIO()
    mask_image.save(byte_arr, format='PNG')

    # Encode the BytesIO object to base64
    base64_str = base64.b64encode(byte_arr.getvalue()).decode('utf-8')

    return base64_str

def scale_image(image_pil, scale_factor):
    '''
    Scale an image proportionally according to its size.
    '''
    # Get the original size
    original_width, original_height = image_pil.size

    # Calculate the new size
    new_width = int(original_width * (1/scale_factor))
    new_height = int(original_height * (1/scale_factor))

    # Resize the image
    scaled_image = image_pil.resize((new_width, new_height))

    return scaled_image

def image_to_base64(image_pil):
    '''
    Convert a PIL image to base64 string.
    '''
    # Save the PIL image to a BytesIO object
    byte_arr = BytesIO()
    image_pil.save(byte_arr, format='PNG')

    # Encode the BytesIO object to base64
    base64_str = base64.b64encode(byte_arr.getvalue()).decode('utf-8')

    return base64_str

def encode_file_to_base64(path):
    with open(path, 'rb') as file:
        return base64.b64encode(file.read()).decode('utf-8')


def url_to_base64(url):
    '''
    Convert an image from a URL to a base64 string.
    '''
    # Send a GET request to the URL
    response = requests.get(url)

    # Open the image
    image = Image.open(BytesIO(response.content))

    # Save the image to a BytesIO object
    byte_arr = BytesIO()
    image.save(byte_arr, format='PNG')

    # Encode the BytesIO object to base64
    base64_str = base64.b64encode(byte_arr.getvalue()).decode('utf-8')

    return base64_str

def read_json_from_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def merge_images(save_path, image_path, body_sam_masks_path):
    # Load the images
    save_image = Image.open(save_path).convert('RGB')
    image = Image.open(image_path).convert('RGB')
    mask_image = Image.open(body_sam_masks_path).convert('RGB')

    # Resize the images to the same size
    max_size = (max(save_image.size[0], image.size[0], mask_image.size[0]),
                max(save_image.size[1], image.size[1], mask_image.size[1]))
    save_image = save_image.resize(max_size)
    image = image.resize(max_size)
    mask_image = mask_image.resize(max_size)

    # Convert the images to numpy arrays
    save_image_array = np.array(save_image)
    image_array = np.array(image)
    mask_array = np.array(mask_image)



    # Create a new array based on the condition
    result_array = np.where(mask_array == 255, save_image_array, image_array)

    # Convert the resulting array back to an image
    result_image = Image.fromarray(result_array.astype(np.uint8))

    # Save the resulting image
    result_path = save_path.replace('.png', '_merged.png')
    result_image.save(result_path)

    return result_path


# ---------------------------------------------------------------------------- #
#                                RunPod Handler                                #
# ---------------------------------------------------------------------------- #
def handler(event):
    '''
    This is the handler function that will be called by the serverless.
    '''
    file_path = './sd_input_template.txt'
    input = event["input"]

    input_json_row = read_json_from_file(file_path)
    input_json = input_json_row['input']


    init_images =   input['image']
    prompt = input['prompt']
    if init_images.startswith('data:image'):
        init_images_base64 = init_images.split(',')[1]
        image_bytes = base64.b64decode(init_images_base64)
        image = Image.open(BytesIO(image_bytes))
        width, height = image.size
        input_json['height'] = height
        input_json['width'] = width
    else:
        init_images_base64 = url_to_base64(init_images)
    image_path = decode_and_save_base64(init_images_base64)

    data = {
        "image_path": image_path,
        "base64": init_images_base64
    }
    start_time = time.time()
    response = requests.post(SAM_URL, json=data)
    response_json = response.json()
    mask_status = response_json['mask_status']
    if mask_status==False:
        return[no_face_url, '']
    mask_image_path = response_json['path']
    white_image_path = response_json['body_cropped_image_path']
    face_average_brightness = response_json['face_average_brightness']
    body_sam_masks_path = response_json['body_sam_masks_path']
    print('white_image_path :'+white_image_path)
    white_image_base64 = encode_file_to_base64(white_image_path)
    average_brightness = response_json['average_brightness']
    print('average_brightness :'+str(average_brightness))
    end_time = time.time()
    print(f"Sam Time taken: {end_time - start_time}")
    mask_base64 =  encode_file_to_base64(mask_image_path)

    if prompt:
        input_json['prompt'] = prompt

    print('prompt :' + input_json['prompt'])
    input_json['mask'] = mask_base64
    #input_json['init_images'] = [init_images_base64]
    input_json['init_images'] = [white_image_base64]
    input_json['alwayson_scripts']['controlnet']['args'][0]['input_image'] = white_image_base64
    input_json['alwayson_scripts']['controlnet']['args'][1]['input_image'] = white_image_base64

    # if(average_brightness > 50):
    #     input_json['inpainting_fill'] = 1
    # else:
    #     input_json['inpainting_fill'] = 0
    input_json['inpainting_fill'] = 0
    print('inpainting_fill :' + str(input_json['inpainting_fill']))
    #input_json['alwayson_scripts']['controlnet']['args'][0]['model'] = 'control_v11p_sd15_openpose [cab727d4]'
    #input_json['alwayson_scripts']['controlnet']['args'][0]['model'] = 'control_v11f1p_sd15_depth [cfd03158]'



    #input_json['alwayson_scripts']['controlnet']['args'][0]['module'] = 'openpose_full'
    start_time = time.time()
    #json = run_inference(event["input"])
    json = run_inference(input_json)
    end_time = time.time()
    print(f"sd  Time taken: {end_time - start_time}")
    for index, image in enumerate(json.get('images')):
        save_path =  decode_and_save_base64(image)
        if index == 0:
            break
    save_path = merge_images(save_path, image_path, body_sam_masks_path)
    #clean_url = upload_image(event['id'], save_path)
    #apply_glass_effect(save_path, 10).save(save_path)
    #blur_url =  upload_image(event['id'],save_path)
    #print(save_path)
    #return [blur_url,clean_url]
    return save_path

def start_serverless_function_in_thread():
    # This function will be run in a separate thread
    runpod.serverless.start({"handler": handler})

def _signal_handler(signalnum, frame):
    print(f"Received signal {signalnum}, exiting...")
    sys.exit(0)
def main():
    '''
    This is the main function that will be called by the serverless.
    '''
    # Check if the service is ready to receive requests
    wait_for_service(url=f'{LOCAL_URL}/txt2img')

    wait_for_service(url=f'{SAM_URL}')
    print("WebUI API Service is ready. Starting RunPod...")

    signal.signal(signal.SIGINT, _signal_handler)

    # Start the serverless function in a separate thread
    thread = threading.Thread(target=start_serverless_function_in_thread)
    thread.start()
    thread.join()


if __name__ == "__main__":
    wait_for_service(url=f'{LOCAL_URL}/txt2img')

    wait_for_service(url=f'{SAM_URL}')
    print("WebUI API Service is ready. Starting RunPod...")

    runpod.serverless.start({"handler": handler})
