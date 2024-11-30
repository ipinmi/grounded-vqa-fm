import pandas as pd
from PIL import Image
import clip, torch
from collections import Counter
import numpy as np

def open_dataset_text(path):
	df = pd.read_json(path, lines=True)
	#print(df.head())
	pattern = df['movie'].str.contains('1054_')
	filtered_df = df[pattern]
	selected_columns = ['img_fn', 'metadata_fn', 'objects', 'question', 'answer_choices', 'answer_label']
	#print(filtered_df[selected_columns][:2].T.to_dict())
	return filtered_df[selected_columns].T.to_dict()

# Get img from img_fn
def get_img(img_path):
	image = Image.open(img_path).convert("RGB")
	return image

# Get bouding box from metadata_fn
def get_object(path):
	df = pd.read_json(path, lines=True)
	selected_columns = ['boxes']
	return df[selected_columns].T.to_dict()

def align_text_and_img():
	data_dict = open_dataset_text('val.jsonl')
	aligned_data_list = []

	for _,entry in data_dict.items():

		img_path = entry['img_fn']
		object_path = entry['metadata_fn']
		
		img = get_img(img_path)
		objects = entry['objects']
		boxes = list(get_object(object_path).values())[0]['boxes']
		question = entry['question']
		answer_choices = entry['answer_choices']
		answer_label =  entry['answer_label']

		# Crop objects in the image
		#cropped_images = []
		#for box in boxes:
		#	x_min, y_min, x_max, y_max, score = box
		#	cropped_image = img.crop((x_min, y_min, x_max, y_max))
		#	cropped_images.append(cropped_image)

		#print(f'objects:{objects}', len(objects))
		#print(f'boxes: {boxes}', len(boxes))
		
		#img.show()
		#for cropped_image in cropped_images:
			#cropped_image.show()
	
		temp_dict = {'img':img, 'boxes':boxes,'objects':objects, 'question':question, 'answer_choices':answer_choices, 'answer_label':answer_label}
		aligned_data_list.append(temp_dict)

	return aligned_data_list

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
				for j,idx in enumerate(token):
					if j==0:
						idx2string += objects[idx]
					else:
						idx2string += 'and'
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

def process_data_entry(data_list):
	res = []
	GENDER_NEUTRAL_NAMES = ['Casey', 'Riley', 'Jessie', 'Jackie', 'Avery', 'Jaime', 'Peyton', 'Kerry', 'Jody', 'Kendall',
                        'Peyton', 'Skyler', 'Frankie', 'Pat', 'Quinn']
	for data_dict in data_list:
		objects = data_dict['objects']
		question = data_dict['question']
		answer_choices = data_dict['answer_choices']
		answer_label = data_dict['answer_label']
		boxes = data_dict['boxes']
		img = data_dict['img']

		cnt = 0
		for i, obj in enumerate(objects):
			if obj == 'person':
				objects[i] = GENDER_NEUTRAL_NAMES[cnt]
				cnt += 1
				cnt = cnt % len(GENDER_NEUTRAL_NAMES)

		instruction = set_instruction(boxes, objects, answer_choices, question)
		question = set_question(objects, question)
		answer_choices = set_answer(objects, answer_choices)

		final_text = []
		for answer in answer_choices:
			template =  ' '.join(question) + ' '.join(answer)
			
			if len(template) > 76:
				template = template[-76:]
			final_text.append(template)

		res.append( (instruction, final_text, img, answer_label) )
	return res

def combine_text_with_img(data_list):
	device = "cuda" if torch.cuda.is_available() else "cpu"
	model, preprocess = clip.load("ViT-B/32", device=device)
	
	y_pred = np.zeros(len(data_list))
	y_true = np.zeros(len(data_list))
	for i, (instruction, texts, img, answer) in enumerate(data_list):
		answers_pred = np.zeros(4)
		for j, text in enumerate(texts):
			#print(text)
			instruction2ids = torch.tensor(clip.tokenize(instruction)).to(device)
			text2ids = torch.tensor(clip.tokenize(text)).to(device)

			image = preprocess(img).unsqueeze(0).to(device)
			with torch.no_grad():
				#instructions_features = model.encode_text(instruction2ids)
				text_features = model.encode_text(text2ids)
				image_features = model.encode_image(image)

				#final_features = (text_features+instructions_features)/2

				sim = (text_features @ image_features.T).squeeze(0)
				answers_pred[j] = sim
		y_pred[i]= answers_pred.argmax()
		y_true[i] = answer
	return y_pred, y_true



#test()
d = align_text_and_img()
data_list = process_data_entry(d)
y_pred, y_true = combine_text_with_img(data_list[:1000])
print(y_pred)
print(y_true)
print((y_pred==y_true).sum()/len(y_pred))

