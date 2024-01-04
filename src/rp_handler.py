import time
import base64
import runpod
import requests
import json
import os
from requests.adapters import HTTPAdapter, Retry
from datetime import datetime
import warnings
from PIL import Image
import base64
import torch
from io import BytesIO
from PIL import ImageChops
import uuid
from runpod.serverless.utils.rp_upload import upload_image
LOCAL_URL = "http://127.0.0.1:3000/sdapi/v1"
SAM_URL = "http://127.0.0.1:8080/process"
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

# ---------------------------------------------------------------------------- #
#                                RunPod Handler                                #
# ---------------------------------------------------------------------------- #
def handler(event):
    '''
    This is the handler function that will be called by the serverless.
    '''
    file_path = '/sd_input_template.txt'
    input = event["input"]

    input_json_row = read_json_from_file(file_path)
    input_json = input_json_row['input']


    init_images =   input['image']
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
        "image_path": image_path
    }
    start_time = time.time()
    response = requests.post(SAM_URL, json=data)
    end_time = time.time()
    print(f"Sam Time taken: {end_time - start_time}")
    mask_image_path = response.text
    mask_base64 =  encode_file_to_base64(mask_image_path)
    input_json['mask'] = mask_base64
    input_json['init_images'] = [init_images_base64]
    input_json['alwayson_scripts']['controlnet']['args'][0]['input_image'] = init_images_base64
    input_json['alwayson_scripts']['controlnet']['args'][0]['model'] = 'control_v11p_sd15_openpose [cab727d4]'
    input_json['alwayson_scripts']['controlnet']['args'][0]['module'] = 'openpose_full'
    start_time = time.time()
    #json = run_inference(event["input"])
    json = run_inference(input_json)
    end_time = time.time()
    print(f"sd  Time taken: {end_time - start_time}")
    for index, image in enumerate(json.get('images')):
        save_path =  decode_and_save_base64(image)
        if index == 0:
            break
    presigned_url =  upload_image(event['id'],save_path)
    print(save_path)
    return presigned_url


if __name__ == "__main__":
    wait_for_service(url=f'{LOCAL_URL}/txt2img')

    print("WebUI API Service is ready. Starting RunPod...")

    runpod.serverless.start({"handler": handler})
