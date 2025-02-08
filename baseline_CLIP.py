import pandas as pd
from dataset import VCRDataset
from dataloader import *
from PIL import Image
import clip, torch
from collections import Counter
import numpy as np


#def align_text_and_img():
#    data_dict = open_dataset_text('val.jsonl')
#    aligned_data_list = []
#
#    for _,entry in data_dict.items():
#
#        fn_path = '../vcr1images/vcr1images/'
#        img_path = fn_path + entry['img_fn']
#        object_path = fn_path + entry['metadata_fn']
#        
#        img = get_img(img_path)
#        objects = entry['objects']
#        boxes = list(get_object(object_path).values())[0]['boxes']
#        question = entry['question']
#        answer_choices = entry['answer_choices']
#        answer_label =  entry['answer_label']
#        rationale_choices = entry['rationale_choices']
#        rationale_label = entry['rationale_label']
#
#        # Crop objects in the image
#        cropped_images = []
#        for box in boxes:
#            x_min, y_min, x_max, y_max, score = box
#            cropped_image = img.crop((x_min, y_min, x_max, y_max))
#            cropped_images.append(cropped_image)
#
#        #print(f'objects:{objects}', len(objects))
#        #print(f'boxes: {boxes}', len(boxes))
#        
#        #img.show()
#        #for cropped_image in cropped_images:
#            #cropped_image.show()
#    
#        temp_dict = {'rationale_choices':rationale_choices, 'rationale_label':rationale_label, 'cropped_images':cropped_images, 'img':img, 'boxes':boxes,'objects':objects, 'question':question, 'answer_choices':answer_choices, 'answer_label':answer_label}
#        aligned_data_list.append(temp_dict)
#        #print('alignment: done')
#    return aligned_data_list



def inference_CLIP(batch_size):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device=device)
    num2coord = [ (i,j) for i in range(4) for j in range(4) ]
    print(f'device:{device}, done')

    dataset = VCRDataset('val.jsonl', img_dir='../vcr1images/vcr1images/')
    list_true_y = []
    list_true_r_y = []
    list_pred_y = []
    list_pred_r_y = []

    for e, batch in enumerate(dataloader(dataset, batch_size)):
        print(f"batch:{e+1}")

        batch = process_batched_data(batch)
        images, QAR_entries, cropped_images, true_y, true_r_y = flatten_question_answers_rationales(batch)
        assert len(images)==len(QAR_entries)
        assert len(QAR_entries[0])==16

        imgs2feat = [ preprocess(image).to(device) for image in images ]
        imgs2feat = torch.stack(imgs2feat).to(device)
        cropped2feat = [ torch.stack(preprocess(img2feat)) for cropped_img_list in cropped_images ]
        
        with torch.no_grad():
	        image_logits = model.encode_image(imgs2feat)

	        #print(imgs2feat.shape)
	        
	        #print('img:', image_logits.shape)
	        #print(QAR_entries[0])

	        for i, flatten_text in enumerate(QAR_entries):
	            QA_list = []
	            for entry in flatten_text:
	                text2ids = [ cut_long_text(text, device) for text in entry ]
	                #print(text2ids)
	                text2ids = torch.cat(text2ids).to(device)
	                text_logits = model.encode_text(text2ids)
	                QA_list.append(text_logits.sum(axis=0)/3)
	            final_text_logits = torch.stack(QA_list)
	            #print(final_text_logits.shape)
	            
	            pred = image_logits[i] @ final_text_logits.T
	            pred = pred.argmax()
	            a, r = num2coord[pred]
	            list_pred_y.append(a)
	            list_pred_r_y.append(r)

	        list_true_y += true_y
	        list_true_r_y += true_r_y


    evaluate(list_pred_y, list_true_y, list_pred_r_y, list_true_r_y)


            
def evaluate(y_pred, y_true, y_r_pred, y_r_true ):
    y_pred, y_true, y_r_pred, y_r_true = np.array(y_pred),np.array(y_true),np.array(y_r_pred),np.array(y_r_true)
    print('ACC for Q->A:')
    print((y_pred==y_true).sum()/len(y_pred))
    print('ACC for QA->R:')
    print((y_r_pred==y_r_true).sum()/len(y_r_pred))

#test

inference_CLIP(batch_size=256)

