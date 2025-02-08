import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image

def open_dataset_text(path):
	df = pd.read_json(path, lines=True)
	#print(df.head())
	pattern = df['movie']
	filtered_df = df #df[pattern]
	selected_columns = ['img_fn', 'metadata_fn', 'objects', 'question', 'answer_choices', 'answer_label', 'rationale_choices', 'rationale_label']
	filtered_df[selected_columns].head().info()
	print('raw data: done')
	return (filtered_df[selected_columns][:10000].T).to_dict()

# Get img from img_fn
def get_img(img_path):
	image = Image.open(img_path).convert("RGB")
	return image

# Get bouding box from metadata_fn
def get_object(path):
	df = pd.read_json(path, lines=True)
	selected_columns = ['boxes']
	return df[selected_columns].T.to_dict()




class VCRDataset(Dataset):
    def __init__(self, dataset_file, img_dir='../vcr1images/vcr1images/', transform=None):
        # 加载数据集
        self.data_dict = open_dataset_text(dataset_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        entry = self.data_dict[idx]

        img_path = self.img_dir + entry['img_fn']
        object_path = self.img_dir + entry['metadata_fn']

        img = get_img(img_path)
        objects = entry['objects']
        boxes = list(get_object(object_path).values())[0]['boxes']
        question = entry['question']
        answer_choices = entry['answer_choices']
        answer_label = entry['answer_label']
        rationale_choices = entry['rationale_choices']
        rationale_label = entry['rationale_label']

        # Crop objects in the image
        cropped_images = []
        for box in boxes:
            x_min, y_min, x_max, y_max, score = box
            cropped_image = img.crop((x_min, y_min, x_max, y_max))
            if self.transform:
                cropped_image = self.transform(cropped_image)
            cropped_images.append(cropped_image)

        # 返回数据（图像、问题、选择、标签等）
        return {
            'rationale_choices': rationale_choices,
            'rationale_label': rationale_label,
            'cropped_images': cropped_images,
            'img': img,
            'boxes': boxes,
            'objects': objects,
            'question': question,
            'answer_choices': answer_choices,
            'answer_label': answer_label
        }



if __name__=='__main__':
    dataset = VCRDataset('val.jsonl', img_dir='../vcr1images/vcr1images/')
    print(dataset[0].keys())
    print(dataset[0]['objects'])
    