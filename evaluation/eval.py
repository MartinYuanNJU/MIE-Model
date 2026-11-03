from collections import defaultdict
from copy import deepcopy
from typing import Dict, List
import sys
import os
import re
import json
import time
import copy
import itertools
import yaml
from tqdm import tqdm

# Allow running as `python evaluation/eval.py` from repo root.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if os.path.dirname(__file__) not in sys.path:
    sys.path.insert(0, os.path.dirname(__file__))

from Predict import *
from Scorer import Metric

Description_EventType = {
    "Movement:Transport":"This event type involves the movement of people or goods from one location to another. The mode of transport can include land vehicles (such as cars, buses, trains), air (such as airplanes), or water (such as ships). The key aspect of this event is the change in location or the displacement",
    "Conflict:Attack": "This event type refers to an aggressive action or assault directed at individuals, groups, or facilities, usually with the intent to cause harm or damage. An attack can involve physical violence, use of weapons, or other forms of forceful engagement.",
    "Conflict:Demonstrate": "This event type involves organized public gatherings, often in the form of protests or rallies, where participants express their opinions, demands, or grievances, usually against authorities or policies. Demonstrations can be peaceful or involve civil unrest.",
    "Justice:Arrest-Jail": "This event type occurs when law enforcement authorities apprehend an individual suspected of a crime, leading to their detention. The process typically includes arrest, booking, and confinement in a jail or prison, pending further legal actions.",
    "Contact:Phone-Write": "This event type involves communication between individuals or groups through phone calls, written correspondence, or electronic messages. It typically includes exchanges of information, instructions, or personal updates via verbal or written means",
    "Contact:Meet": "This event type occurs when two or more individuals come together in person for a discussion, negotiation, or social interaction. Meetings can take place in various settings, such as offices, homes, or public places, and are often planned for specific purposes.",
    "Life:Die": "This event type marks the occurrence of death, signifying the end of an individual's life. It is often associated with related events such as funerals, memorial services, and the emotional responses of those affected by the loss.",
    "Transaction:Transfer-Money": "This event type involves the exchange or movement of money between parties, either within financial institutions, through online platforms, or in person. The transfer of money can occur for various reasons, including payments, remittances, or investments."
}

def load_yaml(yaml_path):
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    return data      

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

yaml_path = os.path.join(os.path.dirname(__file__), "dataset_eval.yaml")
yaml_data = load_yaml(yaml_path)['dataset']
def predict_etype(predict_model, sentence, image_path):
    '''predict the event type'''
    prompt = format_detect.format(sentence)
    etype_pred = predict_model.evaluate_llava(prompt, image_path)
    return etype_pred

def predict_args(predict_model, sentence, image_path, etype, trigger):
    '''predict the event args'''
    event_type = recover_et(etype) if ':' not in etype else etype
    args = get_arg_from_template(event_type)
    args_pred = list()
    for arg in args:
        prompt = f"According to the image and the input text, extract the argument '{arg}' of the event '{etype}'.The trigger word in the text is '{trigger}'. If you can't find Corresponding argument, output 'None'. Use <seg> to separate text argument and image argument.\n\ninput text:{sentence}"
        args_pred.append(predict_model(prompt, image_path))
        
    return args_pred

def obtain_data(file_path):
    if not os.path.isabs(file_path):
        file_path = os.path.normpath(os.path.join(os.path.dirname(__file__), file_path))
    with open(file_path, 'r') as file:
        datas = json.load(file)
    return datas


def write_errors(mode):
    return os.path.join(yaml_data['ERRORS'], mode+".log")


event_types = ["Movement:Transport","Conflict:Attack","Conflict:Demonstrate","Justice:Arrest-Jail","Contact:Phone-Write","Contact:Meet","Life:Die","Transaction:Transfer-Money"]

def recover_et(et):
    if ':' in et:
        return et
        
    for event_type_ in event_types:
        if et.strip() in event_type_:
            return event_type_

    return 'None'

def predict_args_zeroshot(predict_model, sentence, image_path, event_type, trigger):
    pass

def predict_args_meae(ans, etype):
    '''predict the event args'''
    event_type = recover_et(etype)
    args = get_arg_from_template(event_type)
    args_pred = [None] * len(args)
    
    # prompt = f"According to the image and the input text, extract the arguments '{', '.join(get_arg_from_template(event_type))}' of the event '{event_type.split(':')[1]}'.The trigger word in the text is '{trigger}'. If you can't find Corresponding argument, output 'None'. Use <seg> to separate text argument and image argument.\n\ninput text:{sentence}"
    # ans = predict_model(prompt, image_path)
    role_list = ans.split('\n')
    role_list_copy = copy.deepcopy(role_list)
    for arg in args:
        for role_ in role_list_copy:
            if arg in role_:
                args_pred[args.index(arg)] = role_.replace(arg, '').replace(':', '')
                if role_ in role_list:
                    role_list.remove(role_)
    
    ind = 0
    while ind < len(role_list) and None in args_pred:
        try:
            index = role_list.index(None)
            args_pred[index] = role_list[ind]
            ind += 1
        except ValueError:
            break
    
    args_pred = list(map(lambda x: 'None' if x is None else x, args_pred))

    return args_pred

def pipeline_shot(mode):
    etype_metric = Metric()

    # triggers_metric = Metric()
    arguments_metric = Metric()
    predict_event_model = Predict(model_base="xxx/models/llava-v1.5-7b", model_path="xxx/mmee_models/llava_lora_multi_ed", device_id=1)
    predict_model = Predict(model_base='xxx/models/llava-v1.5-7b', model_path='xxx/mmee_models/llava_lora_multi', device_id=2)
    # predict_event_model = Predict(model_base=None, model_path="xxx/models/llava-v1.5-7b", device_id=3)
    # predict_model = predict_event_model
    datas = obtain_data(yaml_data[mode.upper()])

    if mode=="multi":
        for data in datas:
            sentence = data['sentence']
            event_type = data['event_type']
            image_path = os.path.join(yaml_data['IMAGE_DIR'], data['image_id'])
            print(sentence)
            print(image_path)
            
            ans = predict_etype(predict_event_model, sentence, image_path)
            
            if ans =='None':
                et_pred = []
            else:
                et_tmp = [et.strip('()').split(',') for et in ans]
                et_pred = [{'event_type':recover_et(et[0]), 'trigger':et[1].strip() if len(et)==2 else 'None'} for et in et_tmp if recover_et(et[0]) in Description_EventType]
            et_gold = [{'event_type':et['event_type'], 'trigger':et['trigger']['text']} for et in data['text_mention']] 
            etype_metric.count_instance(et_gold, et_pred)

            print(et_pred)
            print(et_gold)
            args_pred_txt, args_pred_img =[], []
            for et_p in et_pred:
                event_type = et_p['event_type'].split(':')[1]
                trigger = et_p['trigger']
                
                args = predict_args(predict_model, sentence, image_path, event_type, trigger)
                
                for arg, role in zip(args, get_arg_from_template(recover_et(event_type))):
                    
                    split_arg = arg.split('<seg>')
                    if len(split_arg) == 2:
                        arg_txt, arg_imgs = split_arg
                    else:
                        arg_txt = arg_imgs = arg

                    if 'None' not in arg_txt:
                        args_pred_txt.append({"text":arg_txt, "role":role, "event_type":event_type})
                    
                    for arg_img in arg_imgs.split('.'):
                        if 'None' in arg_img:
                            continue
                        content = image_path + '\n' + arg_img + '\n'
                        with open('xxx/tmp/llava2seem.txt', 'w') as f:
                            f.write(content)

                        while True:
                            if os.path.exists('xxx/tmp/seem2llava.txt'):
                                with open('xxx/tmp/seem2llava.txt', 'r') as f:
                                    lines = f.readlines()
                                if lines:
                                    if sum(json.loads(lines[0])) < 0:
                                        break
                                    open("xxx/tmp/seem2llava.txt", 'w').close()
                                    args_pred_img.append({"localization":json.loads(lines[0]), "role":role, "event_type":event_type})
                                    break
                            time.sleep(1)  

            args_gold_txt = [{"text":arg['text'], "role":arg['role'], "event_type":data['event_type'].split(':')[1]} for arg in data['text_mention'][0]['arguments']]
            args_gold_img = [{"localization":argument[1:], "role":role, "event_type":data['event_type'].split(':')[1]} for role, arguments in data['image_mention']['role'].items() for argument in arguments]
            
            arguments_metric.count_instance_txt(args_gold_txt, args_pred_txt)
            arguments_metric.count_instance_iou(args_gold_img, args_pred_img)
            # print(args_pred_txt)
            print(args_pred_img)
            # print(args_gold_txt)
            print(args_gold_img)
            print("arguments_metric: ", arguments_metric.compute_f1())
            print("etype_metric: ", etype_metric.compute_f1())

            print()
    else:
        print()
    
    print(etype_metric.compute_f1())
    print(arguments_metric.compute_f1())


def eval_zeroshot(mode):
    etype_metric = Metric()
    # triggers_metric = Metric()
    arguments_metric = Metric()
    # predict_event_model = Predict(model_base="xxx/models/llava-v1.5-7b", model_path="xxx/mmee_models/llava_lora_multi_ed_wotrain", device_id=3)
    # predict_model = Predict(model_base='xxx/models/llava-v1.5-7b', model_path="xxx/mmee_models/llava_lora_multi_train_wotrain", device_id=3)
    predict_event_model = Predict(model_base=None, model_path="xxx/models/llava-v1.5-7b", device_id=3)
    predict_model = predict_event_model
    datas = obtain_data(yaml_data[mode.upper()])

    if mode=="multi":
        for data in datas:
            sentence = data['sentence']
            event_type = data['event_type']
            image_path = os.path.join(yaml_data['IMAGE_DIR'], data['image_id'])
            print(sentence)
            print(image_path)
            def predict_etype_zero(predict_model, sentence, image_path):
                '''predict the event type'''
                format_detect_zero = "According to the image and the input text, extract all the possible events, each event associated with an event type must include a trigger word.If there is no candidate event, output 'None'. This trigger verb must be included in the sentence.\nsentence:{}\n\nCandidate event types: Transport, Attack, Demonstrate, Arrest-Jail, Phone-Write, Meet, Die, Transfer-Money, None.The output format is (event type, trigger word).Use '\n' to separate multiple (event type, trigger word) pairs."
                prompt = format_detect_zero.format(sentence)
                etype_pred = predict_model(prompt, image_path)
                return etype_pred
            ans = predict_etype_zero(predict_event_model, sentence, image_path)
            print(ans)
            if ans =='None':
                et_pred = []
            else:
                et_tmp = [et.strip('()').split(',') for et in ans.split('\n')]
                et_pred = [{'event_type':recover_et(et[0]), 'trigger':et[1].strip(' "\'') if len(et)==2 else 'None'} for et in et_tmp if recover_et(et[0]) in Description_EventType]
            et_gold = [{'event_type':et['event_type'], 'trigger':et['trigger']['text']} for et in data['text_mention']] 
            etype_metric.count_instance(et_gold, et_pred)

            print(et_pred)
            print(et_gold)
            def predict_args_zs(predict_model, sentence, image_path, event_type, trigger):
                event_type = recover_et(event_type)
                args = get_arg_from_template(event_type)
                args_pred = [None] * len(args)
                
                prompt = f"According to the image and the input text, extract the arguments '{', '.join(get_arg_from_template(event_type))}' of the event '{event_type.split(':')[1]}'.The trigger word in the text is '{trigger}'. If you can't find Corresponding argument, output 'None'. Use <seg> to separate text argument and image argument.Use '\n' to separate 'text argument <seg> image argument' pairs.\n\ninput text:{sentence}"
                ans = predict_model(prompt, image_path)
                role_list = ans.split('\n')
                role_list_copy = copy.deepcopy(role_list)
                for arg in args:
                    for role_ in role_list_copy:
                        if arg in role_:
                            args_pred[args.index(arg)] = role_.replace(arg, '').replace(':', '')
                            if role_ in role_list:
                                role_list.remove(role_)
                
                ind = 0
                while ind < len(role_list) and None in args_pred:
                    try:
                        index = role_list.index(None)
                        args_pred[index] = role_list[ind]
                        ind += 1
                    except ValueError:
                        break
                
                args_pred = list(map(lambda x: 'None' if x is None else x, args_pred))

                return args_pred
            
            args_pred_txt, args_pred_img =[], []
            for et_p in et_pred:
                event_type = et_p['event_type'].split(':')[1]
                trigger = et_p['trigger']
                args = predict_args_zs(predict_model, sentence, image_path, event_type, trigger)
                
                for arg, role in zip(args, get_arg_from_template(recover_et(event_type))):
                    
                    split_arg = arg.split('<seg>')
                    if len(split_arg) == 2:
                        arg_txt, arg_imgs = split_arg
                    else:
                        arg_txt = arg_imgs = arg

                    if 'None' not in arg_txt:
                        args_pred_txt.append({"text":arg_txt, "role":role, "event_type":event_type})
                    
                    for arg_img in arg_imgs.split('.'):
                        if 'None' in arg_img:
                            continue
                        content = image_path + '\n' + arg_img + '\n'
                        with open('xxx/tmp/llava2seem.txt', 'w') as f:
                            f.write(content)

                        while True:
                            if os.path.exists('xxx/tmp/seem2llava.txt'):
                                with open('xxx/tmp/seem2llava.txt', 'r') as f:
                                    lines = f.readlines()
                                if lines:
                                    if sum(json.loads(lines[0])) < 0:
                                        break
                                    open("xxx/tmp/seem2llava.txt", 'w').close()
                                    args_pred_img.append({"localization":json.loads(lines[0]), "role":role, "event_type":event_type})
                                    break
                            time.sleep(1)  

            args_gold_txt = [{"text":arg['text'], "role":arg['role'], "event_type":event_type} for arg in data['text_mention'][0]['arguments']]
            args_gold_img = [{"localization":argument[1:], "role":role, "event_type":event_type} for role, arguments in data['image_mention']['role'].items() for argument in arguments]
            
            arguments_metric.count_instance_txt(args_gold_txt, args_pred_txt)
            arguments_metric.count_instance_iou(args_gold_img, args_pred_img)
            print(args_pred_txt)
            print(args_pred_img)
            print(args_gold_txt)
            print(args_gold_img)
            print("arguments_metric: ", arguments_metric.compute_f1())
            print("etype_metric: ", etype_metric.compute_f1())

            print()
    else:
        print()
    
    print(etype_metric.compute_f1())
    print(arguments_metric.compute_f1())
    
if __name__=='__main__':
    pipeline_shot(mode="multi")
    # eval_zeroshot(mode="multi")
    
    