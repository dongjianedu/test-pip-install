from flask import Flask, request, jsonify
from lang_sam import LangSAM
from PIL import Image
from PIL import ImageDraw
from PIL import ImageChops
import torch
import uuid
app = Flask(__name__)
model = LangSAM()

@app.route('/process', methods=['POST'])
def process_image():
    image_path = request.json['image_path']
    image_pil = Image.open(image_path).convert("RGB")
    head_masks, head_boxes, head_phrases, head_logits = model.predict(image_pil, 'head')
    body_masks, body_boxes, body_phrases, body_logits = model.predict(image_pil, 'body')
    head_box = head_boxes[0]
    head_box = head_box.squeeze()
    start_x = head_box[0].item()
    start_y = head_box[1].item()
    end_x = head_box[2].item()
    end_y = head_box[3].item()
    mask = Image.new('L', image_pil.size, 255)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([start_x, start_y, end_x, end_y], fill=0)
    head_box_result=mask
    #head_box_result.save('./head_box_result.png')
    body_mask = body_masks[0]
    body_mask_image = Image.fromarray((body_mask.float() * 255).to(torch.uint8).numpy())
    result_mask_image = ImageChops.darker(head_box_result, body_mask_image)
    random_uuid = uuid.uuid4()
    random_file_name = str(random_uuid) + '.png'
    path = f"/tmp/{random_file_name}"
    result_mask_image.save(path)
    return path

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080,debug=True)