from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split

class TrainValDataset_FeaturesExtracted(Dataset):
    def __init__(self, feature_path, json_file_path, tokeniser, max_sequence_length, padding_idx, train=True, size=20000):
        self.feature_path = feature_path
        self.json_file_path = json_file_path
        self.tokeniser = tokeniser
        self.max_sequence_length = max_sequence_length
        self.padding_idx = padding_idx
        X         = []
        y_reports = []

        labels_df       = pd.read_json(self.json_file_path)

        if (train):
            subset_file = pd.read_json('/netscratch/gsingh/MIMIC_CXR/DataSet/MIMIC_CXR_Reports/Report_CSV_Files/subset_train.json')
        else:
            subset_file = pd.read_json('/netscratch/gsingh/MIMIC_CXR/DataSet/MIMIC_CXR_Reports/Report_CSV_Files/subset_valid.json')
        subset_file_list = []
        for rows, items in subset_file.iterrows():
            subset_file_list.append(str(items.loc['image_name']))
        for rows, items in tqdm(labels_df.iterrows()):
            if str(items.loc['image_name']) in subset_file_list:
                # if train and size < 1:
                #     break
                subset_file_list.remove(str(items.loc['image_name']))
                X.append(os.path.join(self.feature_path, str(items.loc['image_name']) + '.npy'))
                report = np.array(self.tokeniser.encode(items.loc['report'].lower()))
                if(report.shape[0]<self.max_sequence_length):
                    paddings = np.full((self.max_sequence_length-report.shape[0],),self.padding_idx)
                    final_encoded_report = np.concatenate((report,paddings))
                    # final_encoded_report = report
                else:
                    final_encoded_report = report[:self.max_sequence_length]
                y_reports.append(final_encoded_report)
                # size -= 1
        self.x_images   = X
        self.y_reports = y_reports

    def __len__(self):
        return len(self.y_reports)

    def __getitem__(self, idx):
        img_name = self.x_images[idx]
        image     = np.load(img_name)
        y_report = np.array(self.y_reports[idx])
        img_name = img_name.split('/')[-1]
        return img_name, image, y_report

class TrainValDataset_Images(Dataset):
    def __init__(self, image_path, json_file_path, tokeniser, max_sequence_length, padding_idx, image_size=299, train=True, size=20000):
        self.image_path = image_path
        self.json_file_path = json_file_path
        self.tokeniser = tokeniser
        self.max_sequence_length = max_sequence_length
        self.padding_idx = padding_idx
        X         = []
        y_reports = []

        labels_df       = pd.read_json(self.json_file_path)

        if (train):
            subset_file = pd.read_json('/netscratch/gsingh/MIMIC_CXR/DataSet/MIMIC_CXR_Reports/Report_CSV_Files/subset_train.json')
        else:
            subset_file = pd.read_json('/netscratch/gsingh/MIMIC_CXR/DataSet/MIMIC_CXR_Reports/Report_CSV_Files/subset_valid.json')
        subset_file_list = []
        for rows, items in subset_file.iterrows():
            subset_file_list.append(str(items.loc['image_name']))
        for rows, items in tqdm(labels_df.iterrows()):
            if str(items.loc['image_name']) in subset_file_list:
                if train and size < 1:
                    break
                subset_file_list.remove(str(items.loc['image_name']))
                X.append(os.path.join(self.image_path, str(items.loc['image_name'])))
                report = np.array(self.tokeniser.encode(items.loc['report'].lower()))
                if(report.shape[0]<self.max_sequence_length):
                    paddings = np.full((self.max_sequence_length-report.shape[0],),self.padding_idx)
                    final_encoded_report = np.concatenate((report,paddings))
                else:
                    final_encoded_report = report[:self.max_sequence_length]
                y_reports.append(final_encoded_report)
                size -= 1
        self.x_images   = X
        self.y_reports = y_reports

        self.transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor(),
                                             transforms.Normalize((0.4721, 0.4721, 0.4721),
                                                                  (0.2996, 0.2996, 0.2996)),
                                             ])

    def __len__(self):
        return len(self.y_reports)

    def __getitem__(self, idx):
        img_name = self.x_images[idx]
        image = Image.open(img_name).convert('RGB')
        image = self.transform(image)
        y_report = np.array(self.y_reports[idx])
        img_name = img_name.split('/')[-1]
        return img_name, image, y_report


class TestDataset_FeaturesExtracted(Dataset):
    def __init__(self, feature_path, json_file_path, tokeniser, max_sequence_length, padding_idx):
        self.feature_path = feature_path
        self.json_file_path = json_file_path
        self.tokeniser = tokeniser
        self.max_sequence_length = max_sequence_length
        self.padding_idx = padding_idx

        X = []
        y_reports = []

        labels_df = pd.read_json(self.json_file_path)
        for rows, items in tqdm(labels_df.iterrows()):
            X.append(os.path.join(self.feature_path, str(items.loc['image_name']) + '.npy'))
            report = np.array(self.tokeniser.encode(items.loc['report'].lower()))
            if(report.shape[0]<self.max_sequence_length):
                paddings = np.full((self.max_sequence_length-report.shape[0],),self.padding_idx)
                final_encoded_report = np.concatenate((report,paddings))
            else:
                final_encoded_report = report[:self.max_sequence_length]
            y_reports.append(final_encoded_report)

        self.x_images   = X
        self.y_reports = y_reports

    def __len__(self):
        return len(self.y_reports)

    def __getitem__(self, idx):
        img_name = self.x_images[idx]
        image     = np.load(img_name)
        y_report = np.array(self.y_reports[idx])
        img_name = img_name.split('/')[-1]
        return img_name, image, y_report

class TestDataset_Images(Dataset):
    def __init__(self, image_path, json_file_path, tokeniser, max_sequence_length, padding_idx, image_size=299):
        self.image_path = image_path
        self.json_file_path = json_file_path
        self.tokeniser = tokeniser
        self.max_sequence_length = max_sequence_length
        self.padding_idx = padding_idx

        X = []
        y_reports = []
        labels_df = pd.read_json(self.json_file_path)
        for rows, items in tqdm(labels_df.iterrows()):
            X.append(os.path.join(self.image_path, str(items.loc['image_name'])))
            report = np.array(self.tokeniser.encode(items.loc['report'].lower()))
            if(report.shape[0]<self.max_sequence_length):
                paddings = np.full((self.max_sequence_length-report.shape[0],),self.padding_idx)
                final_encoded_report = np.concatenate((report,paddings))
            else:
                final_encoded_report = report[:self.max_sequence_length]
            y_reports.append(final_encoded_report)
        self.x_images   = X
        self.y_reports = y_reports

        self.transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor(),
                                             transforms.Normalize((0.4721, 0.4721, 0.4721),
                                                                  (0.2996, 0.2996, 0.2996)),
                                             ])

    def __len__(self):
        return len(self.y_reports)

    def __getitem__(self, idx):
        img_name = self.x_images[idx]
        image = Image.open(img_name).convert('RGB')
        image = self.transform(image)
        y_report = np.array(self.y_reports[idx])
        img_name = img_name.split('/')[-1]
        return img_name, image, y_report

class DataSet_Indiana(Dataset):
    def __init__(self, image_path, json_file_path, tokeniser, max_sequence_length, padding_idx, sos_idx, eos_idx, image_size=299):
        self.transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor(),
                                             transforms.Normalize((0.4721, 0.4721, 0.4721),
                                                                  (0.2996, 0.2996, 0.2996)),
                                             ])
        df = pd.read_json(json_file_path, orient='index')
        X = []
        y_actual_reports = []
        y_input_reports = []
        for rows, items in df.iterrows():
            X.append(os.path.join(image_path, str(items.loc['image_name'])))
            input_report = np.array(tokeniser.encode_as_ids(items.loc['modified_report']))
            actual_report = np.array(tokeniser.encode_as_ids(items.loc['modified_report']))
            np.concatenate([[sos_idx], input_report])
            np.concatenate([actual_report, [eos_idx]])
            if(input_report.shape[0]<max_sequence_length):
                paddings = np.full((max_sequence_length-input_report.shape[0],),padding_idx)
                final_encoded_input_report = np.concatenate((input_report,paddings))
                final_encoded_actual_report = np.concatenate((actual_report, paddings))

            else:
                final_encoded_input_report = input_report[:max_sequence_length]
                final_encoded_actual_report = actual_report[:max_sequence_length]
                np.concatenate([final_encoded_actual_report, [eos_idx]])
            y_input_reports.append(final_encoded_input_report)
            y_actual_reports.append(final_encoded_actual_report)
        self.X = X
        self.y_input_reports = y_input_reports
        self.y_actual_reports = y_actual_reports

    def __len__(self):
        return len(self.y_actual_reports)

    def __getitem__(self, idx):
        img_name = self.X[idx]
        image = Image.open(img_name).convert('RGB')
        image = self.transform(image)
        y_input_reports = np.array(self.y_input_reports[idx])
        y_actual_reports = np.array(self.y_actual_reports[idx])
        img_name = img_name.split('/')[-1]
        return img_name, image, y_input_reports, y_actual_reports

class Create_DataLoader():
    def __init__(self,image_path, tokeniser=None, transform=None, json_file_path=None, shuffle=False,
                 max_sequence_length=256,sos_idx=2,eos_idx=3, padding_idx=0, batch_size=4, num_workers=0, image_size=299, random_state=42):
        self.image_path = image_path
        self.transform = transform
        self.json_file_path = json_file_path
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokeniser = tokeniser
        self.max_sequence_length = max_sequence_length
        self.padding_idx = padding_idx
        self.image_size = image_size
        self.random_state = random_state
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx

    def train_val_dataloader_features_extracted(self,train=True, size=20000):
        dataset = TrainValDataset_FeaturesExtracted(feature_path=self.image_path, json_file_path=self.json_file_path,
                                                    tokeniser=self.tokeniser, train=train,
                                                    max_sequence_length=self.max_sequence_length,
                                                    padding_idx=self.padding_idx, size=size)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)

    def train_val_dataloader_images(self,train=True, size=20000):
        dataset = TrainValDataset_Images(image_path=self.image_path, json_file_path=self.json_file_path,
                                         tokeniser=self.tokeniser, train=train,
                                         max_sequence_length=self.max_sequence_length,
                                         padding_idx=self.padding_idx, size=size, image_size=self.image_size)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)

    def test_dataloader_features_extracted(self):
        dataset = TestDataset_FeaturesExtracted(image_path=self.image_path,json_file_path=self.json_file_path, tokeniser=self.tokeniser,
                                                max_sequence_length=self.max_sequence_length, padding_idx=self.padding_idx)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)

    def test_dataloader_images(self):
        dataset = TestDataset_Images(image_path=self.image_path,json_file_path=self.json_file_path, tokeniser=self.tokeniser, 
                                     max_sequence_length=self.max_sequence_length, padding_idx=self.padding_idx, image_size=self.image_size)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)

    def create_indiana_dataset(self, train=False):
        dataset = DataSet_Indiana(image_path=self.image_path, json_file_path=self.json_file_path, tokeniser=self.tokeniser,
                                  max_sequence_length=self.max_sequence_length, padding_idx=self.padding_idx, sos_idx=self.sos_idx,
                                  eos_idx=self.eos_idx, image_size=self.image_size)

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=train, num_workers=self.num_workers)