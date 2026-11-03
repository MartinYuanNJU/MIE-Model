import torch
from PIL import Image
# from lavis.models import load_model_and_preprocess
# from transformers import Blip2Processor, Blip2ForConditionalGeneration
from collections import defaultdict
import os
import json
from tqdm import tqdm
from Predict import Predict

device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
seem_prompt = defaultdict(lambda: defaultdict(str))
img_dirs = os.listdir('xxx/swig_images')
with open('xxx/swig-master/SWiG_jsons/train.json', 'r') as file:
    data = json.load(file)

with open('xxx/swig-master/SWiG_jsons/imsitu_space.json', 'r') as file:
    mapping = json.load(file)['nouns']


predict_model = Predict(model_base=None, model_path="xxx/models/llava-v1.5-7b", device_id=0)

# query = "Generate a caption based on the weapon in the image."
# raw_image = Image.open('xxx/datasets/images_512/shooting_135.jpg').convert('RGB')
# inputs = processor(raw_image, query, return_tensors="pt").to("cuda:0", torch.float16)
# out = model.generate(**inputs)
# print(processor.decode(out[0], skip_special_tokens=True).strip())



for img_dir in tqdm(img_dirs):

    sub_dir = os.path.join('xxx/swig_images', img_dir)
    img_paths = os.listdir(sub_dir)
    for img_path in img_paths:
        raw_image = Image.open(os.path.join(sub_dir, img_path)).convert('RGB')
        try:
            noun_id = data[img_dir]['frames'][0][img_path.replace('.jpg', '')]
            if noun_id != '':
                query = f"Describe the {mapping[noun_id]['gloss'][0]} in the image briefly."
            else:
                query = "Describe the image briefly."
            
            out = predict_model(query, os.path.join(sub_dir, img_path))

            seem_prompt[img_dir][img_path.replace('.jpg', '')] = out.replace("in the image ", '').strip()
        except ValueError as e:
            pass

with open('swig_role.json', 'w') as json_file:
    json.dump(seem_prompt, json_file, indent=4)  


