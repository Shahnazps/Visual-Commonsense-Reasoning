import io
import os
import json
from pydantic import BaseModel,Field,FilePath
from opyrator.components.types import FileContent
from PIL import Image 
imagePath = "/home/shahnaz/Documents/academics/main_project/opyrator/opyrator/"

#model imports
import json
from subprocess import run

import sys
sys.path.insert(0, "/media/disk/user/shahnaz/project/r2c")
#import argparse
from dataloaders.vcr import VCR, VCRLoader
import torch
from allennlp.common.params import Params
from torch.nn import DataParallel
import multiprocessing
from allennlp.models import Model
import models
from torch.nn.modules import BatchNorm2d
import numpy as np
from nltk.tokenize import word_tokenize
from config import VCR_ANNOTS_DIR
from dataloaders.vcr import VCR,VCRLoader
from utils.pytorch_misc import time_batch, restore_best_checkpoint
from config import VCR_ANNOTS_DIR

mode = "answer"

split = "val"

folder = '../../saves/flagship_{}'.format(mode)

params = Params.from_file('../../models/multiatt/default.json')
NUM_GPUS = torch.cuda.device_count()
NUM_CPUS = multiprocessing.cpu_count()
if NUM_GPUS == 0:
    raise ValueError("you need gpus!")


def _to_gpu(td):
    if NUM_GPUS > 1:
        return td
    for k in td:
        if k != 'metadata':
            if isinstance(td[k], dict):
                td[k] = {k2: v.cuda(non_blocking=True) for k2, v in td[k].items()}
            else:
                td[k].cuda(non_blocking=True)
    return td


num_workers = (4 * NUM_GPUS if NUM_CPUS == 32 else 2*NUM_GPUS)-1
print(f"Using {num_workers} workers out of {NUM_CPUS} possible", flush=True)
loader_params = {'batch_size': 1, 'num_gpus': 1, 'num_workers': num_workers}

dataset = VCR.custom_splits(split, 0)
model = Model.from_params(vocab=dataset.vocab, params=params['model'])
for submodule in model.detector.backbone.modules():
    if isinstance(submodule, BatchNorm2d):
        submodule.track_running_stats = False
    for p in submodule.parameters():
        p.requires_grad = False
model = DataParallel(model).cuda() if NUM_GPUS > 1 else model.cuda()
restore_best_checkpoint(model, folder)


def eval(dataset, model, index, num):
    dataset_loader = VCRLoader.from_dataset(dataset, **loader_params)
    # print("Loading {} for {}".format(params['model'].get('type', 'WTF?'),
    # 'rationales' if args.rationale else 'answer'), flush=True)
    print("Params ", params.__dict__)
    model.eval()
    val_probs = []
    val_labels = []
    for b, batch in enumerate(dataset_loader):
        print(b)
        print("num : {} {}".format(num, num == 0))
        with torch.no_grad():
            if num == 0:
                if b == index:
                    batch = _to_gpu(batch)
                    output_dict = model(**batch)
                    val_probs.append(output_dict['label_probs'].detach().cpu().numpy())
                    val_labels.append(batch['label'].detach().cpu().numpy())
                    break
            else:
                batch = _to_gpu(batch)
                output_dict = model(**batch)
                val_probs.append(output_dict['label_probs'].detach().cpu().numpy())
                val_labels.append(batch['label'].detach().cpu().numpy())
    print("Val Labels : ", val_labels)
    val_labels = np.concatenate(val_labels, 0)
    val_probs = np.concatenate(val_probs, 0)
    acc = float(np.mean(val_labels == val_probs.argmax(1)))
    return {"label": val_labels, "pred": val_probs.argmax(1), "acc": acc}
    # print("val_labels: {} and val_probs: {}".format(val_labels, val_probs.argmax(1)))
    # print("Final val accuracy is {:.3f}".format(acc))
    # np.save(os.path.join(folder, f'valpreds_custom.npy'), val_probs)

def get_details(index):
    split = "val"
    # dataset = VCR.custom_spluts(split,0)
    sampleJson = {}
    pred_dict = {}
    num = 0
    with open(os.path.join(VCR_ANNOTS_DIR,'{}.jsonl'.format(split)),'r') as f:
        for i,s in enumerate(f):
            if index == i:
                sampleJson = json.loads(s)
                break
    if mode == "rationale":
        conditioned_label = sampleJson["answer_label"]
        sampleJson['question'] += sampleJson['answer_choices'][conditioned_label]
    answer_choices = sampleJson['{}_choices'.format(mode)]
    #rawdata
    sampleJson['index'] = index
    sampleJson['answer_choices'] = answer_choices
    sampleJson['img_path'] = "images/vcr1images/" + sampleJson['img_fn']
    print("answers ",sampleJson['answer_choices'][0])


# def main_page():
#     split = "val"
#     dataset = VCR.custom_splits(split, 0)
#     sampleJson = {}
#     pred_dict = {}
#     num = 0
#     split = "val"
#     if request.method == 'POST':
#         # file = request.files['file']
#         # filename = secure_filename(file.filename)
#         # file.save(os.path.join('uploads', filename))
#         # return redirect(url_for('prediction', filename=filename))
#         # return redirect(url_for('display',filename=filename))
#         if request.form['get_details']:
#             index = int(request.form['index'])
#             with open(os.path.join(VCR_ANNOTS_DIR, '{}.jsonl'.format(split)), 'r') as f:
#                 for i, s in enumerate(f):
#                     if index == i:
#                         sampleJson = json.loads(s)
#                         break
#             if mode == "rationale":
#                 conditioned_label = sampleJson['answer_label']
#                 sampleJson['question'] += sampleJson['answer_choices'][conditioned_label]
#             answer_choices = sampleJson['{}_choices'.format(mode)]
#
#             # rawdata
#             sampleJson['index'] = index
#             # sampleJson['question']=sampleJson['question']
#             sampleJson['answer_choices'] = answer_choices
#             sampleJson['img_path'] = "images/vcr1images/" + sampleJson['img_fn']
#         elif request.form['get_prediction']:
#             sampleJson = json.loads(request.form["raw_data"].replace(
#                 "\\u0027", '"').replace("\"\"\"", "\"'\""))
#             index = int(sampleJson["index"])
#             num = int(request.form['num'])
#             if num == 1:
#                 with open("../data/custom.jsonl", "w") as f:
#                     json.dump(sampleJson, f)
#
#                 # os.system('python data/get_bert_embeddings/extract_features.py --name bert_da --init_checkpoint data/get_bert_embeddings/bert-pretrain/model.ckpt-53230 --split=val_custom')
#                 run('python ../data/get_bert_embeddings/extract_features.py --name bert_da --init_checkpoint ../data/get_bert_embeddings/bert-pretrain/model.ckpt-53230 --split=custom', shell=True)
#
#                 dataset = VCR.custom_splits('custom', index)
#                 split = "custom"
#                 # print("New dataset", new_dataset.__dict__)
#             pred_dict = eval(dataset, model, index, num)
#         elif request.form['customize']:
#             num = 1
#             sampleJson = json.loads(request.form["raw_data"].replace("\\u0027", '"'))
#             if request.form['question']:
#                 question = word_tokenize(request.form['question'])
#                 sampleJson["question"] = question
#             for i in range(4):
#                 text = request.form['answer{}'.format(i+1)]
#                 print("Text : {} answer{}".format(text, i+1))
#                 if text:
#                     sampleJson["answer_choices"][i] = word_tokenize(text)
#             print("Sample json ", sampleJson)
#             if mode == "rationale":
#                 # for i in range(4):
#                 #     # text = input(f"Enter the rationale choice no {i}: ")
#                 #     text = input(f"Enter the rationale choice no {i}\n" +
#                 #                  f"(current rationale for {i})" +
#                 #                  "{sampleJson['rationale_choices'][i]}: ")
#                 #     if text:
#                 #         sampleJson["rationale_choices"][i] = word_tokenize(text)
#                 pass
#     return render_template('index.html', raw_data=sampleJson, pred_dict=pred_dict, num=num)

#
# def main():
#     print("Image Path ", raw_data['img_path'])
#     print("Question ", raw_data['question'])
#     print("Answer Choice ", raw_data['answers'])
#     print("Sample json:", sampleJson)
#
#     # data_instance = {x: None for x in ['movie', 'objects', 'interesting_scores',
#     #                                    'answer_likelihood', 'img_fn', 'metadata_fn', 'answer_orig',
#     #                                    'question_orig', 'rationale_orig', 'question', 'answer_match_iter',
#     #                                    'answer_sources', 'answer_choices', 'answer_label',
#     #                                    'rationale_choices', 'rationale_sources', 'rationale_match_iter',
#     #                                    'rationale_label', 'img_id', 'question_number', 'annot_id',
#     #                                    'match_fold', 'match_index']}
#     # data_instance.update({"question": [], "answer_choices": [], "rationale_choices": []})
#
#     # text = input("Enter the custom question : ")
#     text = input(f"Enter the custom question:\n" +
#                  "(current question) {sampleJson['question']}: ")
#     if text:
#         question = word_tokenize(text)
#         sampleJson["question"] = question
#     for i in range(4):
#         text = input(f"Enter the answer choice no {i}\n" +
#                      f"(current answer for {i})" +
#                      "{sampleJson['answer_choices'][i]}: ")
#         if text:
#             sampleJson["answer_choices"][i] = word_tokenize(text)
#     if args.mode == "rationale":
#         for i in range(4):
#             text = input(f"Enter the rationale choice no {i}\n" +
#                          f"(current rationale for {i})" +
#                          "{sampleJson['rationale_choices'][i]}: ")
#             if text:
#                 sampleJson["rationale_choices"][i] = word_tokenize(text)
#     print("New sample json ", sampleJson)
#
#     # print("New sample json ",sampleJson)
#     # for i in range(4):
#     #     text = input(f"Enter the rationale choice no {i}: ")
#     #     data_instance["answer_choices"].append(word_tokenize(text))
#     #  data_instance["answer_label"] = 0
#     #  data_instance["rationale_label"] = 0
#     #  print("Data instance : ",data_instance)
#     #  with open("data/val_custom.jsonl", "w") as f:
#     #      json.dump(sampleJson, f)
#
#     run('python data/get_bert_embeddings/extract_features.py --name bert_da --init_checkpoint data/get_bert_embeddings/bert-pretrain/model.ckpt-53230 --split=val_custom', shell=True)
#     # os.system('python data/get_bert_embeddings/extract_features.py --name bert_da --init_checkpoint data/get_bert_embeddings/bert-pretrain/model.ckpt-53230 --split=val_custom')
#
#     new_dataset = VCR.custom_splits('custom', index)
#     print("New dataset", new_dataset.__dict__)
#     eval(new_dataset,model,1)
#     #print("New dataset : ",new_dataset[0])
#
#
# if __name__ == '__main__':
#     main()







def loadImage(path,index):
    photo = str(index) + ".jpg"
    path = os.path.join(path,photo)
    
    img = Image.open(path)
    
    img_byte = io.BytesIO()
    img.save(img_byte,format="PNG")
    return img_byte.getvalue()

class ImageNo(BaseModel):
    #image:FileContent = Field(...,mime_type="image/png")
    index:int



class OutputImage(BaseModel):
    image:FileContent = Field(...,mime_type="image/png")
    question:str = Field(
            ...
            )
    answer1: str = Field(
        ...,
        description="Choices for the above question",
        example="He is eating",
        max_length=140,
    )
    answer2: str = Field(
        ...,
        description="Choices for the above question",
        example="He is dancing",
        max_length=140,
    )
    answer3: str = Field(
        ...,
        description="Choices for the above question",
        example="She is sleeping",
        max_length=140,
    )
    answer4: str = Field(
        ...,
        description="Choices for the above question",
        example="The person is cooking",
        max_length=140,
    )

def modelOutput(input:ImageNo)->OutputImage:
    get_details(input.index)
    return OutputImage(image=loadImage(imagePath,input.index),question="",answer1="",answer2="",answer3="",answer4="")
