from dataset import *
import clip, torch

def process_batched_data(data_list):
    res = []
    #GENDER_NEUTRAL_NAMES = ['Casey', 'Riley', 'Jessie', 'Jackie', 'Avery', 'Jaime', 'Peyton', 'Kerry', 'Jody', 'Kendall', 'Peyton', 'Skyler', 'Frankie', 'Pat', 'Quinn']
    
    for data_dict in data_list:

        objects = data_dict['objects']
        question = data_dict['question']
        answer_choices = data_dict['answer_choices']
        answer_label = data_dict['answer_label']
        boxes = data_dict['boxes']
        img = data_dict['img']
        cropped_images = data_dict['cropped_images']
        reason_choices = data_dict['rationale_choices']
        reason_label = data_dict['rationale_label']

        #cnt = 0
        #for i, obj in enumerate(objects):
        #    if obj == 'person':
        #        objects[i] = GENDER_NEUTRAL_NAMES[cnt]
        #        cnt += 1
        #        cnt = cnt % len(GENDER_NEUTRAL_NAMES)

        instruction = set_instruction(boxes, objects, answer_choices, question)
        question = set_question(objects, question)
        answer_choices = set_answer(objects, answer_choices)
        cropped_imgs = set_cropped_img(cropped_images, objects, answer_choices, question)
        reason_choices = set_answer(objects, reason_choices)
        #print(reason_choices)

        QA_R_text = [ ' '.join(choice) for choice in reason_choices ]
        
        final_text = []
        for answer in answer_choices:
            template =  ' '.join(question) +'。'+ ' '.join(answer)
            final_text.append(template)
        #print(QA_R_text)
        res.append( (instruction, final_text, img, answer_label, cropped_imgs, QA_R_text, reason_label ) )
    return res

def flatten_question_answers_rationales(data_list):
    images = []
    cropped_images = []
    QAR_entries = []
    true_y = []
    true_r_y = []
    for i, (instruction, texts, img, answer,cropped_imgs, QA_R_text, QA_R_label) in enumerate(data_list):
        # Make choice from A-D for QA, find (j,k) which has the highest similarity score
        
        #img_ = preprocess(img).to(device)
        #images.append(img_)
        #for cropped_img in cropped_imgs:
        #   cropped_images_features =.append(preprocess(cropped_img).unsqueeze(0).to(device))
        #   image_features += model.encode_image(cropped_img)
        #image_features /= len(cropped_images) + 1

        flatten_QAR = []
        true_y.append(answer)
        true_r_y.append(QA_R_label)
        images.append(img)
        cropped_images.append(cropped_imgs)
        for j, text in enumerate(texts): 
            q,a = text.split('。')
            
            for k, reason in enumerate(QA_R_text):
                flatten_QAR.append([instruction+q, a, reason])

        QAR_entries.append(flatten_QAR)
    
    return images, cropped_images, QAR_entries, true_y, true_r_y

def dataloader(dataset, batch_size = 128):
    for i in range(0, len(dataset), batch_size):
        yield [ dataset[i+k] for k in range(batch_size) if i+k<len(dataset)]

def set_question(objects, question):
    # Map index to objects. e.g. 0-> person, add index to object name
    # objects = [ obj+'_'+str(i) for i,obj in enumerate(objects) ]
    
    # Convert index to object names
    for i,token in enumerate(question):
        idx2string = ''
        if type(token) == type([]):
            for j,idx in enumerate(token):
                if j==0:
                    idx2string += objects[idx]
                else:
                    idx2string += 'and'
                    idx2string += objects[idx]
            question[i] = idx2string

    return question

def set_answer(objects, answers):
    #objects = [ obj+'_'+str(i) for i,obj in enumerate(objects) ]
    for answer in answers:
        for i,token in enumerate(answer):
            idx2string = ' '
            if type(token) == type([]):
                for j, idx in enumerate(token):
                    if j==0:
                        idx2string += objects[idx]
                    else:
                        idx2string += ' and '
                        idx2string += objects[idx]
                answer[i] = idx2string
    return answers

def set_instruction(boxes, objects, answers, question):
    # Detect what objects are related to question
    ids = set()
    for token in question:
        if type(token) == type([]):
            ids.update(token)

    for answer in answers:
        for token in answer:
            if type(token) == type([]):
                ids.update(token)

    # Prompt instruction
    instruction = 'There is '
    for idx in ids:
        box, obj = boxes[idx], objects[idx]
        x,y,m,n, _ = box
        prompt = f'a {obj} in {x:.0f} {y:.0f} {m:.0f} {n:.0f}'
        instruction += prompt
    instruction += 'of the picture.'
    return instruction[:77]

def set_cropped_img(cropped_images, objects, answers, question):
    # Detect what objects are related to question
    ids = set()
    for token in question:
        if type(token) == type([]):
            ids.update(token)

    for answer in answers:
        for token in answer:
            if type(token) == type([]):
                ids.update(token)

    cropped_imgs_filtered = []
    for idx in ids:
        img, obj = boxes[cropped_images], objects[idx]
        cropped_imgs_filtered.append(img)

    return cropped_imgs_filtered


def cut_long_text(text, device):
    p = len(text)
    while p>=0:
        try:
            text2ids = clip.tokenize(text[:p]).clone().detach().to(device)
            break
        except:
            p-=1
    else:
        text2ids = clip.tokenize(text[:77]).clone().detach().to(device)
    return text2ids

if __name__=='__main__':
    dataset = VCRDataset('val.jsonl', img_dir='../vcr1images/vcr1images/')
    #print(dataset[0])
    for batch in dataloader(dataset, batch_size = 128):
        batch = process_batched_data(batch)
        images, QAR_entries, cropped_imgs, true_y, true_r_y = flatten_question_answers_rationales(batch)
        assert len(images)==len(QAR_entries)
        assert len(QAR_entries[0])==16
        print(QAR_entries[0])
        break