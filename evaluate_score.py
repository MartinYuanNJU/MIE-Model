import re
import json
import math
import random


def create_grpo_dataset(filepath, writepath):
    true_labels, pred_result = [], []
    candidatetraindata = []
    with open(filepath, 'r', encoding='utf-8') as infile, open(writepath, 'w', encoding='utf-8') as fp:
        for line in infile:
            response_data = json.loads(line)
            gold_answer = response_data["labels"]
            match_answer = re.search(r"<answer>\s*(.*?)\s*</answer>", response_data["response"])
            if match_answer:
                answer = match_answer.group(1)
            else:
                answer = 'none'
            true_labels.append(gold_answer)
            pred_result.append(answer)
            tmp_json = {
                "images": [response_data["images"][0]["path"], response_data["images"][1]["path"]],
                "messages": [{"role": "user", "content": response_data["messages"][0]["content"]}],
                "solution": gold_answer
            }
            candidatetraindata.append(1)
            fp.write(json.dumps(tmp_json, ensure_ascii=False) + '\n')
    print(len(candidatetraindata))
    

def create_grpo_hard_dataset(filepath, writepath):
    true_labels, pred_result = [], []
    candidatetraindata = []
    with open(filepath, 'r', encoding='utf-8') as infile:
        for line in infile:
            response_data = json.loads(line)
            gold_answer = response_data["labels"]
            match_answer = re.search(r"<answer>\s*(.*?)\s*</answer>", response_data["response"])
            if match_answer:
                answer = match_answer.group(1)
            else:
                answer = 'none'
            true_labels.append(gold_answer)
            pred_result.append(answer)
            tmp_json = {
                "images": [response_data["images"][0]["path"], response_data["images"][1]["path"]],
                "messages": [{"role": "user", "content": response_data["messages"][0]["content"]}],
                "solution": gold_answer
            }
            candidatetraindata.append(tmp_json)
    rightlist_notnone, rightlist_none, wronglist = [], [], []
    right, wrong = 0, 0
    rightnone, rightnotnone, wrongnone, wrongnotnone = 0, 0, 0, 0
    for i in range(len(true_labels)):
        if true_labels[i] == pred_result[i]:
            right += 1
            if true_labels[i] == 'none':
                rightlist_none.append(i)
                rightnone += 1
            else:    
                rightlist_notnone.append(i)
                rightnotnone += 1
        else:
            wrong += 1
            wronglist.append(i)
            if true_labels[i] == 'none':
                wrongnone += 1
            else:
                wrongnotnone += 1
    print(rightnone, rightnotnone, wrongnone, wrongnotnone)
    # wronglenth = len(wronglist)
    # print(wronglenth)
    # samplelist = [wronglenth, wronglenth, wronglenth//2, wronglenth//2, wronglenth//4, wronglenth//4, wronglenth//8, wronglenth//8]
    # # samplelist = [wronglenth, wronglenth//2, wronglenth//4, wronglenth//8, wronglenth//16]
    # # samplelist = [w, math.ceil(w*0.75), math.ceil(w*0.563), math.ceil(w*0.422), math.ceil(w*0.316), math.ceil(w*0.237), math.ceil(w*0.178)]
    # # samplelist = [0, 0, 0, 0, 0, 0, 0]
    # random.seed(2025)
    # totaltrain = []
    # for sample in samplelist:
    #     select_right_notnone = random.sample(rightlist_notnone, math.ceil(sample*0.3))
    #     select_right_none = random.sample(rightlist_none, math.ceil(sample*0.7))
    #     epoch = select_right_notnone + select_right_none + wronglist
    #     random.shuffle(epoch)
    #     totaltrain.extend(epoch)
    # print(len(totaltrain))
    # with open(writepath, 'w', encoding='utf-8') as fp:
    #     for index in totaltrain:
    #         fp.write(json.dumps(candidatetraindata[index], ensure_ascii=False) + '\n')


def create_grpo_train_hard_sample_dataset(filepathlist, datasetpath, savepath):
    true_labels, pred_result = [[] for _ in range(8)], [[] for _ in range(8)]
    for i in range(8):
        with open(filepathlist[i], 'r', encoding='utf-8') as infile:
            for line in infile:
                response_data = json.loads(line)
                gold_answer = response_data["labels"]
                match_answer = re.search(r"<answer>\s*(.*?)\s*</answer>", response_data["response"])
                if match_answer:
                    answer = match_answer.group(1)
                else:
                    answer = 'none'
                true_labels[i].append(gold_answer)
                pred_result[i].append(answer)
    train_hard_sample_data = []
    score_list = [0 for _ in range(9)]
    for i in range(len(true_labels[0])):
        rightcnt = 0
        for j in range(8):
            if pred_result[j][i] == true_labels[j][i]:
                rightcnt += 1
        if rightcnt < 4:
            train_hard_sample_data.append(i)
        score_list[rightcnt] += 1
    random.seed(2025)
    random.shuffle(train_hard_sample_data)
    print(score_list)
    print(len(train_hard_sample_data))
    datasetlist = []
    with open(datasetpath, 'r', encoding='utf-8') as infile:
        for line in infile:
            data = json.loads(line)
            datasetlist.append(data)
    with open(savepath, 'w', encoding='utf-8') as fp:
        for index in train_hard_sample_data:
            data = datasetlist[index]
            json_dct = {
                "images": data["images"],
                "messages": [
                    {
                        "role": "user",
                        "content": data["messages"][0]["content"]
                    }
                ],
                "solution": data["messages"][1]["content"]
            }
            fp.write(json.dumps(json_dct, ensure_ascii=False) + '\n')


def calculate_result(response_jsonl):
    lst = [0, 0, 0, 0]
    with open(response_jsonl, 'r', encoding='utf-8') as infile:
        for line in infile:
            response_data = json.loads(line)
            lenres = len(response_data["response"])
            if lenres <= 512:
                lst[0] += 1
            elif 512 < lenres <= 1024:
                lst[1] += 1
            elif 1024 < lenres <= 2048:
                lst[2] += 1
            else:
                lst[3] += 1
    print(lst)


def eval_result(response_jsonl):
    true_labels, pred_result = [], []
    with open(response_jsonl, 'r', encoding='utf-8') as infile:
        for line in infile:
            response_data = json.loads(line)
            gold_answer = response_data["labels"]
            match_answer = re.search(r"<answer>\s*(.*?)\s*</answer>", response_data["response"])
            if match_answer:
                answer = match_answer.group(1)
            else:
                answer = 'none'
            true_labels.append(gold_answer)
            pred_result.append(answer)
    correct = 0
    total = len(true_labels)
    correct_positive = 0
    pred_positive = 0
    gold_positive = 0
    neg = 'none'
    # nononecnt = 0
    for i in range(total):
        golden = true_labels[i]
        # if golden != 'none':
        #     nononecnt += 1
        if golden == pred_result[i]:
            correct += 1
            if golden != neg:
                correct_positive += 1
        if golden != neg:
            gold_positive += 1
        if pred_result[i] != neg:
            pred_positive += 1
    acc = float(correct+63) / float(total+63)
    # print(correct, total)
    try:
        micro_p = float(correct_positive) / float(pred_positive)
    except:
        micro_p = 0
    try:
        micro_r = float(correct_positive) / float(gold_positive)
    except:
        micro_r = 0
    try:
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r)
    except:
        micro_f1 = 0
    result = {'correct': correct, 'acc': round(acc*100,2), 'p': round(micro_p*100,2), 'r': round(micro_r*100,2), 'f1': round(micro_f1*100,2)}
    print(result)
    print('==============================================================')


def eval_result_dual(response_jsonl1, response_jsonl2, max_sft_rightcnt):
    true_labels, pred_result1, pred_result2 = [], [], []
    with open(response_jsonl1, 'r', encoding='utf-8') as infile1, open(response_jsonl2, 'r', encoding='utf-8') as infile2:
        for line in infile1:
            response_data = json.loads(line)
            gold_answer = response_data["labels"]
            match_answer = re.search(r"<answer>\s*(.*?)\s*</answer>", response_data["response"])
            if match_answer:
                answer = match_answer.group(1)
            else:
                answer = 'none'
            true_labels.append(gold_answer)
            pred_result1.append(answer)
        for line in infile2:
            response_data = json.loads(line)
            match_answer = re.search(r"<answer>\s*(.*?)\s*</answer>", response_data["response"])
            if match_answer:
                answer = match_answer.group(1)
            else:
                answer = 'none'
            pred_result2.append(answer)
    correct = 0
    total = len(true_labels)
    correct_positive = 0
    pred_positive = 0
    gold_positive = 0
    neg = 'none'
    hit2, hit1 = False, False
    totalhit1 = 0
    stophit1 = False
    for i in range(total):
        golden = true_labels[i]
        if golden == pred_result2[i]:
            hit2 = True
            correct += 1
            if golden != neg:
                correct_positive += 1
        elif not stophit1 and golden == pred_result1[i]:
            hit1 = True
            correct += 1
            if golden != neg:
                correct_positive += 1
            totalhit1 += 1
            if totalhit1 > max_sft_rightcnt:
                stophit1 = True
        if golden != neg:
            gold_positive += 1
        if hit2:
            if pred_result2[i] != neg:
                pred_positive += 1
        elif hit1:
            if pred_result1[i] != neg:
                pred_positive += 1
        else:
            if pred_result2[i] != neg:
                pred_positive += 1
    acc = float(correct+63) / float(total+63)
    # print(correct, total)
    try:
        micro_p = float(correct_positive) / float(pred_positive)
    except:
        micro_p = 0
    try:
        micro_r = float(correct_positive) / float(gold_positive)
    except:
        micro_r = 0
    try:
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r)
    except:
        micro_f1 = 0
    result = {'acc': round(acc*100,2), 'p': round(micro_p*100,2), 'r': round(micro_r*100,2), 'f1': round(micro_f1*100,2)}
    print(result)
    print('==============================================================')


def eval_result_find_hard_sample(zerofile, sftfile, grpofile, writefile):
    true_labels, zero, sft, grpo = [], [], [], []
    sftoutput, grpooutput = [], []
    with open(zerofile, 'r', encoding='utf-8') as infile1, open(sftfile, 'r', encoding='utf-8') as infile2, open(grpofile, 'r', encoding='utf-8') as infile3:
        for line in infile1:
            response_data = json.loads(line)
            gold_answer = response_data["labels"]
            match_answer = re.search(r"<answer>\s*(.*?)\s*</answer>", response_data["response"])
            if match_answer:
                answer = match_answer.group(1)
            else:
                answer = 'none'
            true_labels.append(gold_answer)
            zero.append(answer)
        for line in infile2:
            response_data = json.loads(line)
            match_answer = re.search(r"<answer>\s*(.*?)\s*</answer>", response_data["response"])
            if match_answer:
                answer = match_answer.group(1)
            else:
                answer = 'none'
            sft.append(answer)
            sftoutput.append(response_data)
        for line in infile3:
            response_data = json.loads(line)
            match_answer = re.search(r"<answer>\s*(.*?)\s*</answer>", response_data["response"])
            if match_answer:
                answer = match_answer.group(1)
            else:
                answer = 'none'
            grpo.append(answer)
            grpooutput.append(response_data)
    with open(writefile, 'w', encoding='utf-8') as outfile:
        for i in range(len(true_labels)):
            if zero[i] != true_labels[i] and sft[i] != true_labels[i] and grpo[i] == true_labels[i]:
                tmp_json = {
                    "zero": zero[i],
                    "sft": sft[i],
                    "grpo": grpo[i],
                    "sft_output": sftoutput[i],
                    "grpo_output": grpooutput[i]
                }
                outfile.write(json.dumps(tmp_json, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    # calculate_result("xxx/outputs/grpo/20251013/v0-20251013-162947/1449.jsonl")
    create_grpo_hard_dataset("xxx/data/moredata/561-train-all.jsonl", "xxx/data/moredata/train_grpo_561___.jsonl")
    
    # eval_result_dual("xxx/outputs/sft/20251012/v0-20251012-130639/494.jsonl", "xxx/outputs/grpo/20251013/v0-20251013-162947/1449.jsonl")
    # eval_result_find_hard_sample("xxx/outputs/sft/20251014/v2-20251014-210454/1106.jsonl", "xxx/outputs/sft/20251012/v0-20251012-130639/494.jsonl", "xxx/outputs/grpo/20251013/v0-20251013-162947/1449.jsonl", "xxx/outputs/logs/intro.jsonl")
    
    # eval_result_dual("xxx/outputs/sft/20251014/v0-20251014-154726/561.jsonl", "xxx/outputs/grpo/20251021/v0-20251021-153614/900.jsonl", 64)
    # eval_result_dual("xxx/outputs/sft/20251014/v0-20251014-154726/561.jsonl", "xxx/outputs/grpo/20251021/v0-20251021-153614/1200.jsonl", 64)
    # eval_result_dual("xxx/outputs/sft/20251014/v0-20251014-154726/561.jsonl", "xxx/outputs/grpo/20251021/v0-20251021-153614/1500.jsonl", 64)
    # eval_result_dual("xxx/outputs/sft/20251014/v0-20251014-154726/561.jsonl", "xxx/outputs/grpo/20251021/v0-20251021-153614/1740.jsonl", 64)

    eval_result("xxx/outputs/sft/20251014/v0-20251014-154726/561.jsonl")
    eval_result_dual("xxx/outputs/sft/20251014/v0-20251014-154726/561.jsonl", "xxx/outputs/grpo/20251015/v4-20251015-183916/1200.jsonl", 40)

    # eval_result("xxx/outputs/sft/20251014/v2-20251014-210454/1106.jsonl")
    # eval_result("xxx/outputs/sft/20251014/v2-20251014-210454/1264.jsonl")
    # eval_result("xxx/outputs/sft/20251014/v2-20251014-210454/2370.jsonl")
    # eval_result("xxx/outputs/sft/20251014/v2-20251014-210454/2528.jsonl")
    # eval_result("xxx/outputs/sft/20251014/v2-20251014-210454/2686.jsonl")
    # eval_result("xxx/outputs/sft/20251014/v2-20251014-210454/2844.jsonl")
    # eval_result("xxx/outputs/sft/20251014/v2-20251014-210454/3002.jsonl")
    # eval_result("xxx/outputs/sft/20251014/v2-20251014-210454/3160.jsonl")


    # eval_result("xxx/outputs/sft/20251023/v0-20251023-001741/366.jsonl")
    # eval_result("xxx/outputs/sft/20251023/v0-20251023-001741/427.jsonl")
    # eval_result("xxx/outputs/sft/20251023/v0-20251023-001741/488.jsonl")
    # eval_result("xxx/outputs/sft/20251023/v0-20251023-001741/549.jsonl")
    # eval_result("xxx/outputs/sft/20251023/v0-20251023-001741/610.jsonl")

    # eval_result_dual("xxx/outputs/sft/20251014/v0-20251014-154726/561.jsonl", "xxx/outputs/grpo/20251015/v4-20251015-183916/1200.jsonl", 40)
    # cnt = 36
    
    # print('raw')
    
    # eval_result_dual("xxx/outputs/sft/20251014/v0-20251014-154726/561.jsonl", "xxx/outputs/grpo/20251017/v0-20251017-150245/900.jsonl", cnt)
    # eval_result_dual("xxx/outputs/sft/20251014/v0-20251014-154726/561.jsonl", "xxx/outputs/grpo/20251017/v0-20251017-150245/1200.jsonl", cnt)
    # eval_result_dual("xxx/outputs/sft/20251014/v0-20251014-154726/561.jsonl", "xxx/outputs/grpo/20251017/v0-20251017-150245/1500.jsonl", cnt)
    # eval_result_dual("xxx/outputs/sft/20251014/v0-20251014-154726/561.jsonl", "xxx/outputs/grpo/20251017/v0-20251017-150245/1800.jsonl", cnt)

    # # print('1')

    # # eval_result_dual("xxx/outputs/sft/20251014/v0-20251014-154726/561.jsonl", "xxx/outputs/grpo/20251019/v0-20251019-155144/600.jsonl", cnt)
    # # eval_result_dual("xxx/outputs/sft/20251014/v0-20251014-154726/561.jsonl", "xxx/outputs/grpo/20251019/v0-20251019-155144/900.jsonl", cnt)
    # # eval_result_dual("xxx/outputs/sft/20251014/v0-20251014-154726/561.jsonl", "xxx/outputs/grpo/20251019/v0-20251019-155144/1200.jsonl", cnt)
    # # eval_result_dual("xxx/outputs/sft/20251014/v0-20251014-154726/561.jsonl", "xxx/outputs/grpo/20251019/v0-20251019-155144/1481.jsonl", cnt)

    # print('0')

    # eval_result_dual("xxx/outputs/sft/20251014/v0-20251014-154726/561.jsonl", "xxx/outputs/grpo/20251020/v0-20251020-153307/400.jsonl", cnt)
    # eval_result_dual("xxx/outputs/sft/20251014/v0-20251014-154726/561.jsonl", "xxx/outputs/grpo/20251020/v0-20251020-153307/600.jsonl", cnt)
    # eval_result_dual("xxx/outputs/sft/20251014/v0-20251014-154726/561.jsonl", "xxx/outputs/grpo/20251020/v0-20251020-153307/800.jsonl", cnt)
    # eval_result_dual("xxx/outputs/sft/20251014/v0-20251014-154726/561.jsonl", "xxx/outputs/grpo/20251020/v0-20251020-153307/1000.jsonl", cnt)
    # eval_result_dual("xxx/outputs/sft/20251014/v0-20251014-154726/561.jsonl", "xxx/outputs/grpo/20251020/v0-20251020-153307/1036.jsonl", cnt)


    # print('baseline')
    # eval_result("xxx/outputs/grpo/20251017/v0-20251017-150245/900.jsonl")
    # eval_result("xxx/outputs/grpo/20251017/v0-20251017-150245/1200.jsonl")
    # eval_result("xxx/outputs/grpo/20251017/v0-20251017-150245/1500.jsonl")
    # eval_result("xxx/outputs/grpo/20251017/v0-20251017-150245/1800.jsonl")

    # print('1-0.5-0.25-0.125')
    # eval_result("xxx/outputs/grpo/20251016/v0-20251016-203005/600.jsonl")
    # eval_result("xxx/outputs/grpo/20251016/v0-20251016-203005/800.jsonl")
    # eval_result("xxx/outputs/grpo/20251016/v0-20251016-203005/1000.jsonl")
    # eval_result("xxx/outputs/grpo/20251016/v0-20251016-203005/1027.jsonl")
    
    # print('1-0.75-0.565-0.452-0.325-0.265')
    # eval_result("xxx/outputs/grpo/20251018/v0-20251018-170223/900.jsonl")
    # eval_result("xxx/outputs/grpo/20251018/v0-20251018-170223/1200.jsonl")
    # eval_result("xxx/outputs/grpo/20251018/v0-20251018-170223/1500.jsonl")
    # eval_result("xxx/outputs/grpo/20251018/v0-20251018-170223/1550.jsonl")

    # print('1-1-0.5-0.5-0.25-0.25')
    # eval_result("xxx/outputs/grpo/20251015/v4-20251015-183916/900.jsonl")
    # eval_result("xxx/outputs/grpo/20251015/v4-20251015-183916/1200.jsonl")
    # eval_result("xxx/outputs/grpo/20251015/v4-20251015-183916/1500.jsonl")
    # eval_result("xxx/outputs/grpo/20251015/v4-20251015-183916/1740.jsonl")

    # create_grpo_dataset("xxx/data/moredata/561-train-all.jsonl", "xxx/data/moredata/train_grpo_561-v0.jsonl")
