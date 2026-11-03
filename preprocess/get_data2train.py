import json
from PIL import Image
import re
import yaml
import torch.utils.data as data
from collections import defaultdict
import codecs
import os
import sys
sys.path.append('../')  
# from llava.eval.Predict import *

Event_Template = {
    "Movement:Transport": "<Agent>  <Artifact> in <Vehicle> from <Origin> to <Destination>.",
    "Conflict:Attack": "<Attacker>  <Target> using <Instrument> at <Place>.",
    "Conflict:Demonstrate": "<Entity>  besides <Police> using <Instrument> at <Place>.",
    "Justice:Arrest-Jail": "<Person> was  by <Agent> using <Instrument> at <Place>.",
    "Contact:Phone-Write": "<Entity>  using <Instrument> at <Place>.",
    "Contact:Meet": "<Entity> at <Place>.",
    "Life:Die": "<Agent>  <Victim> with <Instrument> at <Place>.",
    "Transaction:Transfer-Money": "<Giver> <Recipient> with <Money> for the benefit of <Beneficiary> at <Place>."
}

eventtype='"Movement:Transport","Conflict:Attack","Conflict:Demonstrate,"Justice:Arrest-Jail","Contact:Phone-Write","Contact:Meet","Life:Die","Transaction:Transfer-Money"'

def get_arg_from_template(event_type):
    matches = re.findall(r"<(.*?)>", Event_Template[event_type])
    return matches

event_types = ["Movement:Transport","Conflict:Attack","Conflict:Demonstrate","Justice:Arrest-Jail","Contact:Phone-Write","Contact:Meet","Life:Die","Transaction:Transfer-Money"]

def recover_et(et):
    if ':' in et:
        return et
        
    for event_type_ in event_types:
        if et.strip() in event_type_:
            return event_type_

    return 'None'

def load_yaml(yaml_path):
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    return data      


yaml_path = "dataset.yaml"
yaml_data = load_yaml(yaml_path)
yaml_dict = yaml_data['dataset']


def crop_and_save_image(input_path, output_path, coordinate):
    image = Image.open(input_path)
    cropped_image = image.crop(coordinate)
    cropped_image.save(output_path)

def fill_template(enent_type:str, event_mention:dict)->str:
    """ Fill the templte to get ACE2005 labeled data"""
    template = Event_Template[enent_type]
    # matches = re.findall(r'\[(.*?)\]', template)
    tmp = {}
    for argument in event_mention["arguments"]:
        if argument["role"] in tmp:
            tmp[argument["role"]].append(f"<{argument['role']}: {argument['text']}>")
        else:
            tmp[argument["role"]] = [f"<{argument['role']}: {argument['text']}>"]
    for key,val in tmp.items():
        template = template.replace(f"<{key}>", " ".join(val))
    template = template.replace("<Trigger>", f'<Trigger: {event_mention["trigger"]["text"]}>')
    return template

def fill_template_swig(enent_type:str, event_mention:dict):
    """ Fill the templte to get swig labeled data"""
    template = Event_Template[enent_type]
    for key,val in event_mention['arguments'].items():
        if isinstance(val, list):
            tmp =  [f"<ios><{ele}><ioe>"  for ele in val]
            template = template.replace(f"<{key}>", " ".join(tmp))
        else:
            template = template.replace(f"<{key}>", f'<ios><{val}><ioe>')
    return template


class SWIG_Dataset(data.Dataset):
    def __init__(self, swig_ontology_file, trian_json, verb_mapping_file, image_dir):
        swig_info = json.load(open(swig_ontology_file))
        self.nouns = swig_info["nouns"]
        self.verbs = swig_info["verbs"]
        self.image_dir = image_dir
        self.annotation = json.load(open(trian_json))
        self.sr_mapping_verb, self.sr_mapping_role = self.load_mapping_all(verb_mapping_file)
        self.etype_imgs = defaultdict(list)

    def load_mapping_all(self, verb_mapping_file):
        sr_verb_mapping = defaultdict()
        sr_role_mapping = defaultdict(lambda : defaultdict(str))
        for line in codecs.open(verb_mapping_file, 'r', encoding='utf-8'):
            line = line.rstrip('\n')
            tabs = line.split('\t')
            sr_verb = tabs[0]
            sr_role = tabs[1]
            ee_event = tabs[2]
            ee_role = tabs[3]
            sr_role_mapping[sr_verb][sr_role] = ee_role
            sr_verb_mapping[sr_verb] = ee_event.replace('||', ':').replace('|', '-')
        return sr_verb_mapping, sr_role_mapping

    def get_etype_imgs(self):
        for verb_img, img_dict in self.annotation.items():
            
            verb = img_dict['verb'] 
            if verb not in self.sr_mapping_verb:
                self.etype_imgs['None'].append(verb_img)
                continue
            self.etype_imgs[self.sr_mapping_verb[verb]].append(verb_img)
            
        with open('xxx/clip/data/etype_imgs.json', 'w') as file:
            json.dump(self.etype_imgs, file, indent=4)   



import glob
import random
from tqdm import tqdm

def find_corresponding_img(text, img_dir='xxx/ACE-2005-image/train_image'):
    xml_files = glob.glob(f"{img_dir}/*.xml")
    for xml_file in xml_files:
        if(find_in_file(xml_file, text.replace("\n", " "))):
            imgdir = xml_file.replace('.xml', '')
            imgs = glob.glob(f"{imgdir}/*.jpeg")
            if len(imgs)!=0:
                return random.choice(imgs)
    return "xxx/datasets/images_512/0.jpg"

def find_in_file(file_path, target_string):
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if target_string in line:
                return True
    return False

def IoU(boxA, boxB):
    # order = xyxy
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def isCorrect(bbox_annot, bbox_pred, iou_thr=.5):
    iou_value = IoU(bbox_pred, bbox_annot)
    if iou_value >= iou_thr:
        return 1, iou_value
    else:
        return 0, iou_value

def compute_f1():
    tp = 0
    gold = 0
    pred = 0
    with open('train.json', 'r') as file:
        data = json.load(file)

    with open('xxx/SEEM/seem_bbox.json', 'r') as file:
        seem_bbox = json.load(file)

    for image_id, img_dict in tqdm(data.items()):
        
        for key, role_box in img_dict['bb'].items():
            if role_box == [-1, -1, -1, -1]:
                continue
            box_pred = seem_bbox[image_id].get(key, None)
            if box_pred:
                cc = isCorrect(role_box, box_pred)[0]
                tp += cc
            gold += 1
            print(tp/gold)

class JSONReader:
    def __init__(self, input_file_path=None, output_file_path=None):
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path
        self.data = None
        self.id = 1
        self.ace_data = self.read_json_file(yaml_dict['ACE_TRAIN_RAW'])
        self.schema_list = list()
        if input_file_path != None:
            self.data = self.read_json_file(self.input_file_path)

    def read_json_file(self, input_file_path):
        with open(input_file_path, 'r') as file:
            data = json.load(file)
        return data

    def save_file(self, data, output_file_path):
        with open(output_file_path, 'w') as file:
            json.dump(data, file, indent=4)   

    
    def preprocess_ACE2005(self):
        for data_ in tqdm(self.read_json_file(yaml_dict['ACE_TRAIN_RAW'])):
            events = []
            for event_mention in data_["golden-event-mentions"]:
                if event_mention["event_type"] not in Event_Template.keys():
                    continue
                events.append(f"({event_mention['event_type'].split(':', 1)[1]},{event_mention['trigger']['text']})")
            if any([True if event_mention["event_type"] in Event_Template.keys() else False for event_mention in data_["golden-event-mentions"]]):
                self.ace_data.append(data_)
        print(len(self.ace_data))
            # schema = {}
            # schema['id'] = str(self.id)
            # self.id += 1
            # schema['image'] = find_corresponding_img(data_['sentence'])
            
            # # schema['arguments'] = {arg['role']:arg['text']  for arg in event_mention['arguments']}
                
            # schema['conversations'] = [
            #     {
            #         "from": "human",
            #         "value": f'''According to the image and the input text, extract all the possible events, Each event associated with an event type must have a trigger verb.\ninput text: {data_['sentence']}\n\nCandidate event types: Transport, Attack, Demonstrate, Arrest-Jail, Phone-Write, Meet, Die, Transfer-Money, None.'''
            #     },
            #     {
            #         "from": "gpt",
            #         "value":  '\n'.join(events) if len(events)!=0 else 'None'
            #             # "value": "ss"
            #     }
                    
            # ]

            # self.schema_list.append(schema)
            

    def preprocess_swig(self):
        # self.preprocess_ACE2005()
        swig_dataset = SWIG_Dataset(yaml_dict['SWIG_ONTOLOGY'], yaml_dict['SWIG_TRAIN_RAW'], verb_mapping_file="xxx/m2e2/data/ace/ace_sr_mapping.txt", image_dir=yaml_dict['SWIG_IMAGE_DIR'])
        
        verb_num = defaultdict(int)
        
        for image_id, img_dict in tqdm(swig_dataset.annotation.items()):
            verb = img_dict['verb'] 
            if verb not in swig_dataset.sr_mapping_verb:
                verb_num[verb] += 1
                if verb_num.get(verb, 0)>=20:
                    continue
                event_type = 'None'
            else:
                event_type = swig_dataset.sr_mapping_verb[verb].split(':')[1]
            
            schema = {}
            schema['id'] = image_id
            schema['image'] = os.path.join(swig_dataset.image_dir, image_id)
            
            schema['conversations'] = [
                {
                    "from": "human",
                    "value": f"According to the image, extract the possible event from candidate event types.\nCandidate event types: Transport, Attack, Demonstrate, Arrest-Jail, Phone-Write, Meet, Die, Transfer-Money, None."
                },
                {
                    "from": "gpt",
                    "value": event_type
                }
            ]       
            self.schema_list.append(schema)
        
    
    def preprocess_et(self):
        # self.preprocess_ACE2005()
        
        swig_dataset = SWIG_Dataset(yaml_dict['SWIG_ONTOLOGY'], yaml_dict['SWIG_TRAIN_RAW'], verb_mapping_file="xxx/m2e2/data/ace/ace_sr_mapping.txt", image_dir=yaml_dict['SWIG_IMAGE_DIR'])
        with open('xxx/clip/data/tmp_random.json', 'r') as json_file:
            pairs = json.load(json_file)

        def find_cor_img(pairs_, sentence):
            for pair in pairs_:
                if pair['text'] == sentence:
                    return pair['image']
        
        for index, data_ in enumerate(self.ace_data):
            events = []
            event_type = 'None'
            for event_mention in data_["golden-event-mentions"]:
                if event_mention["event_type"] not in Event_Template.keys():
                    continue
                if event_type == 'None':
                    event_type = event_mention['event_type']
                events.append(f"({event_mention['event_type'].split(':', 1)[1]}, {event_mention['trigger']['text']})")
                
            sentence = data_['sentence']
            img_name = find_cor_img(pairs[event_type], sentence)
            img_path = os.path.join('xxx/datasets/images_512', img_name)
            if len(events) == 0:
                continue
            # schema = {}
            # schema['id'] = img_name
            # schema['image'] = img_path
            # schema['conversations'] = [
            #     {
            #         "from": "human",
            #         "value": f'''According to the image and the input text, extract all the possible events, Each event associated with an event type must include a trigger verb. This trigger verb must be included in the input text.\ninput text:{sentence}\n\nCandidate event types: Transport, Attack, Demonstrate, Arrest-Jail, Phone-Write, Meet, Die, Transfer-Money, None.'''
            #     },
            #     {
            #         "from": "gpt",
            #         "value": '\n'.join(events) if len(events)!=0 else 'None'
                    
            #     }
            # ]
            schema_txt = {}
            schema_txt['id'] = index
            schema_txt['image'] = 'xxx/datasets/images_512/0.jpg'
            schema_txt['conversations'] = [
                {
                    "from": "human",
                    "value": f'''According to the image and the input text, extract all the possible events, each event associated with an event type must include a trigger word. This trigger word must be included in the input text.\ninput text:{sentence}\n\nCandidate event types: Transport, Attack, Demonstrate, Arrest-Jail, Phone-Write, Meet, Die, Transfer-Money, None.'''
                },
                {
                    "from": "gpt",
                    "value": '\n'.join(events) if len(events)!=0 else 'None'
                    
                }
            ]
            self.schema_list.append(schema_txt)

            schema_img = {}
            schema_img['id'] = img_name
            schema_img['image'] = img_path
            schema_img['conversations'] = [
                {
                    "from": "human",
                    "value": f'''According to the image and the input text, extract all the possible events, Each event associated with an event type must include a trigger word.\n\nCandidate event types: Transport, Attack, Demonstrate, Arrest-Jail, Phone-Write, Meet, Die, Transfer-Money, None.'''
                },
                {
                    "from": "gpt",
                    "value": f"({event_type.split(':', 1)[1]}, {img_name.split('_')[0]})"
                    
                }
            ]
            self.schema_list.append(schema_img)
        
        print(len(self.schema_list))

    def preprocess_multi_train(self):
        # self.preprocess_ACE2005()
        
        swig_dataset = SWIG_Dataset(yaml_dict['SWIG_ONTOLOGY'], yaml_dict['SWIG_TRAIN_RAW'], verb_mapping_file="xxx/m2e2/data/ace/ace_sr_mapping.txt", image_dir=yaml_dict['SWIG_IMAGE_DIR'])
        
        with open('xxx/LAVIS/swig_role_1.json', 'r') as json_file:
            seem_prompt = json.load(json_file)
        
        with open('xxx/clip/data/tmp.json', 'r') as json_file:
            pairs = json.load(json_file)

        def find_cor_img(pairs_, sentence):
            for pair in pairs_:
                if pair['text'] == sentence:
                    return pair['image']

        for index, data_ in enumerate(self.ace_data):
            # event_all = {}
            img_path = ""
            sentence = data_['sentence']

            no_events = True
            for event_mention in data_["golden-event-mentions"]:
                et = event_mention['event_type']
                if et not in Event_Template:
                    continue
                
                event_type = et
                no_events = False
                img_name = find_cor_img(pairs[event_type], sentence)
                img_path = os.path.join('xxx/datasets/images_512', img_name)
                
                arg_c = defaultdict(list)
                if not seem_prompt.get(img_name, None):
                    continue
                for arg, caption in seem_prompt[img_name].items():
                    arg_c[swig_dataset.sr_mapping_role[img_name.split('_')[0]][arg]].append(caption)

                arguments = {arg['role']:arg['text'] for arg in event_mention['arguments']}
                # event_all[f"{event_type.split(':', 1)[1]}, {event_mention['trigger']['text']}"] = '\n'.join([role + ':' + arguments.get(role, 'None')  + ' <seg>' + '.'.join(arg_c.get(role, ['None'])) for role in get_arg_from_template(event_type)])
                # conversations = [
                #     ({
                #         "from": "human",
                #         "value": f'''According to the image and the input text, extract the argument '{role}' of the event '{event_type.split(":", 1)[1]}'.The trigger word in the text is '{event_mention['trigger']['text']}'. If you can't find Corresponding argument, output 'None'. Use <seg> to separate text argument and image argument.\n\ninput text:{sentence}'''
                #     },
                #     {
                #         "from": "gpt",
                #         "value": arguments.get(role, 'None')  + ' <seg> ' + '.'.join(arg_c.get(role, ['None']))
                        
                #     })
                #     for role in get_arg_from_template(event_type)
                # ]
            
            if no_events:
                continue
            
            schema_txt = {}
            schema_txt['id'] = index
            schema_txt['image'] = 'xxx/datasets/images_512/0.jpg'
            conversations = [
                ({
                    "from": "human",
                    "value": f'''According to the image and the input text, extract the argument '{role}' of the event '{event_type.split(":", 1)[1]}'.The trigger word in the text is '{event_mention['trigger']['text']}'. If you can't find Corresponding argument, output 'None'.\n\ninput text:{sentence}'''
                },
                {
                    "from": "gpt",
                    "value": arguments.get(role, 'None')  
                     
                })
                for role in get_arg_from_template(event_type)
            ]
            schema_txt['conversations'] = [item for conversation in conversations for item in conversation]
            self.schema_list.append(schema_txt)

            schema_img = {}
            schema_img['id'] = img_name
            schema_img['image'] = img_path
            conversations = [
                ({
                    "from": "human",
                    "value": f'''According to the image and the input text, extract the argument '{role}' of the event '{event_type.split(":", 1)[1]}'.The trigger word in the text is '{img_name.split("_")[0]}'. If you can't find Corresponding argument, output 'None'.'''
                },
                {
                    "from": "gpt",
                    "value": '.'.join(arg_c.get(role, ['None']))
                    
                })
                for role in get_arg_from_template(event_type)
            ]
            schema_img['conversations'] = [item for conversation in conversations for item in conversation]
            self.schema_list.append(schema_img)

    def preproces_m2e2_multimodal(self):
        datas = []
        text_datas_l, image_datas =  self.read_json_file(yaml_dict['M2E2_TEXT_MULTI']), self.read_json_file(yaml_dict['M2E2_IMAGE_MULTI'])
        text_datas = {text_data['sentence_id']:text_data for text_data in text_datas_l} 

        with open(yaml_dict['M2E2_CROSSMEDIA'], 'r') as file:
            for line in file:
                text_id, image_id, event_type = line.strip().split('\t')  
                image_id.replace(".jpg", "")
                record = {
                    "sentence_id": text_id,
                    "sentence": text_datas[text_id]['sentence'],
                    "image_id": image_id,
                    "text_mention": text_datas[text_id]['golden-event-mentions'],
                    "image_mention": image_datas[image_id.replace(".jpg", "")],
                    "event_type": event_type
                    # "text_mention": text_data['golden-event-mentions']
                }
                datas.append(record)
        self.save_file(datas, yaml_dict['M2E2_MULTI'])
                
    def get_image_traindata(self):
        image_dir = "xxx/datasets/m2e2_rawdata/image/image"
        for image_name, roles in self.data.items():
            schema = {}
            schema['id'] = image_name
            schema['image'] = f"{image_dir}/{image_name}.jpg"
            image = Image.open(f"{image_dir}/{image_name}.jpg")
            width, height = image.size

            pairs = list()
            for entity_name, coordinates in roles['role'].items():
                for i in range(len(coordinates)):
                    x1 = "{:.3f}".format(coordinates[i][1]/width)
                    y1 = "{:.3f}".format(coordinates[i][2]/height)
                    x2 = "{:.3f}".format(coordinates[i][3]/width)
                    y2 = "{:.3f}".format(coordinates[i][4]/height)
                    pairs.append(f"{entity_name}: [{x1}, {y1}, {x2}, {y2}]")
            schema['conversations'] = [
                {
                    "from": "human",
                    "value": "Give you the image, identify its event type"
                },
                {
                    "from": "gpt",
                    "value":  roles["event_type"]
                },
                {
                    "from": "human",
                    "value": f'Give you the image ,event type is {roles["event_type"]}, please extract the corresponding argument type and its coordinates(Left Top and Right Bottom) in the picture '
                },
                {
                    "from": "gpt",
                    "value":  ", ".join(pairs)
                }
            ]            
            self.schema_list.append(schema)
        with open(self.output_file_path, 'w') as file:
            json.dump(self.schema_list, file, indent=4)
        
def merge_json(file1, file2, final_path):
    with open(file1, 'r') as file:
        data1 = json.load(file)
    with open(file2, 'r') as file:
        data2 = json.load(file)
    merge_data = data1 + data2
    with open(final_path, 'w') as file:
        json.dump(merge_data, file, indent=4)   


if __name__ == '__main__':
    # swig_dataset = SWIG_Dataset(yaml_dict['SWIG_ONTOLOGY'], yaml_dict['SWIG_TRAIN_RAW'], verb_mapping_file="xxx/m2e2/data/ace/ace_sr_mapping.txt", image_dir=yaml_dict['SWIG_IMAGE_DIR'])
    # swig_dataset.get_etype_imgs()

    jsonReader = JSONReader('xxx/datasets/SWiG_jsons/train.json')
    jsonReader.preprocess_multi_train()
    jsonReader.save_file(jsonReader.schema_list, 'data/multi_train_unilabel.json')

    

