from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.utils import disable_torch_init
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_PLACEHOLDER
from llava.mm_utils import process_images, tokenizer_image_token, KeywordsStoppingCriteria
import re
from llava.conversation import conv_templates, SeparatorStyle
from PIL import Image
import requests
from io import BytesIO
import torch
import json
from tqdm import tqdm
from peft import PeftModel

Event_Template = {
    "Movement:Transport": "[Movement:Transport]<Agent> <Trigger> <Artifact> in <Vehicle> from <Origin> to <Destination>.",
    "Conflict:Attack": "[Conflict:Attack]<Attacker> <Trigger> <Target> using <Instrument> at <Place>.",
    "Conflict:Demonstrate": "[Conflict:Demonstrate]<Entity> <Trigger> besides <Police> using <Instrument> at <Place>.",
    "Justice:Arrest-Jail": "[Justice:Arrest-Jail]<Person> was <Trigger> by <Agent> using <Instrument> at <Place>.",
    "Contact:Phone-Write": "[Contact:Phone-Write]<Entity> <Trigger> using <Instrument> at <Place>.",
    "Contact:Meet": "[Contact:Meet]<Entity> <Trigger> at <Place>.",
    "Life:Die": "[Life:Die]<Agent> <Trigger> <Victim> with <Instrument> at <Place>.",
    "Transaction:Transfer-Money": "[Transaction:Transfer-Money]<Giver> <Trigger> <Recipient> with <Money> for the benefit of <Beneficiary> at <Place>."
}

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out



class PromptModel:
    def __init__(self, base_model, adapter_path, tokenizer, image_processor, context_len):
        model_path = "xxx/models/llava-v1.5-7b"
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model_name = get_model_name_from_path(model_path)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.context_len = context_len
        self.conv_mode = None
        self.sep = ","
        self.temperature = 0.2
        self.top_p = 0.8
        self.num_beams = 1
        self.max_new_tokens = 2048

    def __call__(self, query, image_file):

        disable_torch_init()
    
        virtual_token_embeddings = self.model.prompt_encoder['default'].embedding
        # print(virtual_token_embeddings.weight.shape)  
        qs = query
        
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in qs:
            if self.model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if self.model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        
        if "llama-2" in self.model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in self.model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        if self.conv_mode is not None and conv_mode != self.conv_mode:
            print(
                "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                    conv_mode, self.conv_mode, self.conv_mode
                )
            )
        else:
            self.conv_mode = conv_mode
        
        conv = conv_templates[self.conv_mode].copy()
        
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
        input_ids = input_ids.to(self.model.device)
        
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        image_files = image_file.split(self.sep)
        images = load_images(image_files)
        images_tensor = process_images(images, self.image_processor, self.model.config).to(self.model.device, dtype=torch.float16)
        input_ids = torch.abs(input_ids)
        # print(model)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids=input_ids,
                images=images_tensor,
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                top_p=self.top_p,
                num_beams=self.num_beams,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(
                f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
            )
        
        outputs = self.tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True
        )[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()
        
        return outputs


class Predict:
    def __init__(self, model_path, model_base, loader=None, device_id=3):
        self.loader = loader
        self.model_path = model_path
        self.model_base = model_base
        self.conv_mode = None
        self.query = \
        "Given the sentence:'US Republican presidential candidate Donald Trump greets supporters at his campaign rally at Werner Enterprises Hangar in Omaha, Nebraska, US May 6, 2016'.Fill in this template: '[txt][Conflict:Attack]<Attacker> <Trigger> <Target> using <Instrument> at <Place>.  <seg> [img][Conflict:Attack]<Attacker> <Trigger> <Target> using <Instrument> at <Place>.'"
        self.sep = ","
        # self.image_file = "xxx/paddlejob/workspace/MMIE/datasets/test/news.jpg"
        self.image_file = \
        "xxx/datasets/images_512/0.jpg"
        # self.offload_folder = "xxx/paddlejob/workspace/MMIE/models/llava-v1.5-7b-offload"
        # self.offload_folder = "xxx/autodl-tmp/models/llava-v1.5-7b-offload"
        self.device = 'cuda:{}'.format(device_id) if torch.cuda.is_available() else 'cpu'
        self.temperature = 0.2
        self.top_p = 0.8
        self.num_beams = 1
        self.max_new_tokens = 2048
        self.model_name, self.tokenizer, self.model, self.image_processor, self.context_len = self.load_model()
        self.model.eval()
        
    def load_model(self):
        model_name = get_model_name_from_path(self.model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(self.model_path, self.model_base, model_name, device=self.device)
        return model_name,tokenizer, model, image_processor, context_len
    
        
    def __call__(self, qs, image_file):
        return self.evaluate_llava(qs, image_file)
         
    def evaluate_llava(self, qs, image_file):

        disable_torch_init()
        # tokenizer.add_tokens([grounding_start, grounding_end], special_tokens=True)
        
        
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in qs:
            if self.model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if self.model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        if "llama-2" in self.model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in self.model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        if self.conv_mode is not None and conv_mode != self.conv_mode:
            print(
                "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                    conv_mode, self.conv_mode, self.conv_mode
                )
            )
        else:
            self.conv_mode = conv_mode

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
        input_ids = input_ids.to(self.model.device)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        image_files = image_file.split(self.sep)
        images = load_images(image_files)
        images_tensor = process_images(images, self.image_processor, self.model.config).to(self.model.device, dtype=torch.float16)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images_tensor,
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                top_p=self.top_p,
                num_beams=self.num_beams,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(
                f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
            )
        outputs = self.tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True
        )[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()
       
        return outputs
    
def save_file(save_path, content):
    with open(save_path, 'w') as file:
        json.dump(content, file, indent=4)


eventtype='"Movement:Transport","Conflict:Attack","Conflict:Demonstrate","Justice:Arrest-Jail","Contact:Phone-Write","Contact:Meet","Life:Die","Transaction:Transfer-Money"'
format_detect = "According to the image and the input text, extract all the possible events, Each event associated with an event type must include a trigger verb. This trigger verb must be included in the input text.\ninput text:{}\n\nCandidate event types: Transport, Attack, Demonstrate, Arrest-Jail, Phone-Write, Meet, Die, Transfer-Money, None."
format_detect_zs = "According to the image and the input text, extract all the possible events, each event associated with an event type must include a trigger word.If there is no candidate event, output 'None'. This trigger verb must be included in the sentence.\nsentence:{}\n\nCandidate event types: Transport, Attack, Demonstrate, Arrest-Jail, Phone-Write, Meet, Die, Transfer-Money, None.The output format is (event type, trigger word)."

if __name__ == '__main__':
    from DataSet import *
    from Scorer import *

    file_path = "xxx/datasets/m2e2_rawdata/m2e2_annotations/text_only_event.json"
    predict = Predict(model_base="xxx/models/llava-v1.5-7b", model_path="xxx/models/llava_lora_eventtype")
    predict_base = Predict(model_base=None, model_path="xxx/models/llava-v1.5-7b")
    with open(file_path, 'r') as file:
        datas = json.load(file)
    etype_metric = Metric()
    etype_base_metric = Metric()
    for data in tqdm(datas):
        if len(data['golden-event-mentions']) == 0:
            continue
        image_file = "xxx/datasets/images_512/0.jpg"
        sentence = data['sentence']
        # detectprompt=''' Your task is to detect an event and its corresponding trigger in the provided multimodal data.The input includes a text description,an image and some important knowledge you should know about this task, the text is [{}], you should pay more attention to the part before 'after'.Use all available information across different modalities to accurately detect events and present the output in the following format: [Event Type : [it must be one of {}]]'''.format(sentence, eventtype)
        prompt = "Your task is to detect an event and its corresponding the provided multimodal data.The input includes a text '{}' ignore the image. Just sent the output in the following format: Event Type : [it must be one of \"Movement:Transport\",\"Conflict:Attack\",\"Conflict:Demonstrate,\"Justice:Arrest-Jail\",\"Contact:Phone-Write\",\"Contact:Meet\",\"Life:Die\",\"Transaction:Transfer-Money\"].".format(sentence)
        gold_eventtype = [mention['event_type'] for mention in data['golden-event-mentions']]
        event_type_pred = predict.evaluate_llava(prompt, image_file).strip()
        etype_metric.count_instance(gold_eventtype, [event_type_pred])
        print(event_type_pred, "----------", gold_eventtype)

        event_type_pred = predict_base.evaluate_llava(prompt, image_file).replace("Event Type:", "").strip()
        etype_base_metric.count_instance(gold_eventtype, [event_type_pred])
        # if "Demonstrate" == event_type_pred:
        #     event_type_pred = "Conflict:Demonstrate"
        # if "Meet" == event_type_pred:
        #     event_type_pred = "Contact:Meet"
        # if data['event_type'] == "Life:Die":
        #     event_type_pred = "Life:Die"
        print(event_type_pred, "----------", gold_eventtype)


    print(etype_metric.compute_f1())
    print(etype_base_metric.compute_f1())
    # for data in test_loader:
        # print(data)
