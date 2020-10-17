from dataloader import Create_DataLoader
import torch
import time
import os
import numpy as np
from report_gen_indiana import ReportGeneration
from config_indiana import Config
import utilities as util
import sentencepiece as sp
import warnings
warnings.filterwarnings("ignore")

training_start_time = time.time()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_C = Config()
image_model, visual_feature_size, image_size = _C.MODELS[0][_C.MODEL_NAME]

tokenizer = sp.SentencePieceProcessor()
tokenizer.load(_C.VOCAB_MODEL_PATH)

if(not _C.SAVED_DATASET):
    train_dataloader = Create_DataLoader(image_path=_C.IMAGE_PATH, json_file_path=_C.TRAIN_JSON_PATH, shuffle=True,
                                            max_sequence_length=_C.MAX_SEQUENCE_LENGTH, batch_size=_C.BATCH_SIZE, image_size=image_size, sos_idx=_C.SOS_INDEX, eos_idx=_C.EOS_INDEX,
                                            padding_idx=_C.PADDING_INDEX, tokeniser=tokenizer, random_state=_C.RANDOM_SEED).create_indiana_dataset(train=True)

    val_dataloader = Create_DataLoader(image_path=_C.IMAGE_PATH, json_file_path=_C.VAL_JSON_PATH, shuffle=True,
                                            max_sequence_length=_C.MAX_SEQUENCE_LENGTH, batch_size=_C.BATCH_SIZE, image_size=image_size, sos_idx=_C.SOS_INDEX, eos_idx=_C.EOS_INDEX,
                                            padding_idx=_C.PADDING_INDEX, tokeniser=tokenizer, random_state=_C.RANDOM_SEED).create_indiana_dataset()

    test_dataloader = Create_DataLoader(image_path=_C.IMAGE_PATH, json_file_path=_C.TEST_JSON_PATH, shuffle=True,
                                            max_sequence_length=_C.MAX_SEQUENCE_LENGTH, batch_size=1, image_size=image_size, sos_idx=_C.SOS_INDEX, eos_idx=_C.EOS_INDEX,
                                            padding_idx=_C.PADDING_INDEX, tokeniser=tokenizer, random_state=_C.RANDOM_SEED).create_indiana_dataset()

    torch.save(train_dataloader, _C.SAVED_DATASET_PATH_TRAIN)
    torch.save(val_dataloader, _C.SAVED_DATASET_PATH_VAL)
    torch.save(test_dataloader, _C.SAVED_DATASET_PATH_TEST)
else:
    train_dataloader = torch.load(_C.SAVED_DATASET_PATH_TRAIN)
    val_dataloader = torch.load(_C.SAVED_DATASET_PATH_VAL)
    test_dataloader = torch.load(_C.SAVED_DATASET_PATH_TEST)

training_loss = []
validation_loss = []


model = ReportGeneration(device=device, image_model=image_model, visual_feature_size=visual_feature_size, max_sequence_length=_C.MAX_SEQUENCE_LENGTH,
                         sos_index=_C.SOS_INDEX, eos_index=_C.EOS_INDEX, embedding_dim=_C.EMBEDDING_DIM, vocab_size=_C.VOCAB_SIZE,
                         num_layers=_C.COMBINED_N_LAYERS, attention_heads=int(_C.EMBEDDING_DIM/_C.D_HEAD), drop_out=_C.DROPOUT_RATE,
                         padding_idx=_C.PADDING_INDEX, number_of_classes=_C.NUM_LABELS, use_beam_search=_C.USE_BEAM_SEARCH, beam_size=_C.BEAM_SIZE)
model.to(device)
torch.set_grad_enabled(True)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=_C.LR_COMBINED, weight_decay=_C.WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=_C.MAX_LR, epochs=_C.EPOCHS, steps_per_epoch=len(train_dataloader))
print('Length of Training data: ',len(train_dataloader))
print('Length of Validation data: ',len(val_dataloader))
print('Length of Test data: ',len(test_dataloader))
print('Starting Training')
from tqdm import tqdm
for epoch in range(_C.EPOCHS):
    start_time = time.time()
    batch_train_loss = []
    batch_val_loss = []
    for batch_train, data_train in enumerate(train_dataloader):
        train_image_name, train_image, train_input_report, train_actual_report = data_train
        report_length = torch.Tensor(np.full((train_input_report.size(0),),_C.MAX_SEQUENCE_LENGTH))
        # if(not _C.EXTRACTED_FEATURES):
        #     train_image = image_model(train_image.to(device))
        train_image = train_image.to(device)
        train_input_report = train_input_report.to(device)
        train_actual_report = train_actual_report.to(device)
        report_length = report_length.to(device)
        optimizer.zero_grad()
        input = {'images':train_image, 'inference': _C.INFERENCE_TIME,
                 'tokenised_report':train_input_report, 'actual_report':train_actual_report, 'report_length':report_length}
        output_dic = model(input)
        train_loss, train_loss_components = output_dic['loss'], output_dic['loss_components']
        # print(type(train_loss), type(train_loss_components))
        batch_train_loss.append(train_loss_components['report_loss'].to('cpu').item())
        train_loss.backward()
        optimizer.step()

        if(batch_train%20==0 and batch_train!=0):
            print('\nCurrently training Batch ',batch_train,' for epoch', epoch, ' and loss for current batch is: ', batch_train_loss[-1])
            print()
    torch.set_grad_enabled(False)
    model.eval()
    print('Inside validation')
    for batch_val, data_val in enumerate(val_dataloader):
        val_image_name, val_image, val_input_report, val_actual_report = data_val
        val_report_length = torch.Tensor(np.full((val_input_report.size(0),),_C.MAX_SEQUENCE_LENGTH))
        # if(not _C.EXTRACTED_FEATURES):
        #     val_image = image_model(val_image.to(device))
        val_image = val_image.to(device)
        val_input_report = val_input_report.to(device)
        val_actual_report = val_actual_report.to(device)
        val_report_length = val_report_length.to(device)
        intersection_list = list(set(val_image_name) & set(_C.VALID_IMAGES))
        val_input = {'images':val_image, 'inference': _C.INFERENCE_TIME, 'tokenised_report':val_input_report, 'actual_report':val_actual_report,
                'report_length':val_report_length}
        val_output_dic = model(val_input)
        val_loss, val_loss_components, predictions = val_output_dic['loss'], val_output_dic['loss_components'],\
                                                     val_output_dic['predictions']
        # print('Loss and Loss components: ', val_loss, val_loss_components)
        predictions = val_output_dic['predictions']
        # print(predictions)
        # for i in range(predictions.shape[0]):
        #     output_report = predictions[i,:]
        #     generated_report = tokenizer.decode(list(filter(lambda a: a != 0, output_report.tolist())))
        #     print('Predicted Report: \n')
        #     print(generated_report)
        if(len(intersection_list)>0):
            for images in intersection_list:
                index = val_image_name.index(images)
                print('\nVal image name: ',val_image_name[index])
                print('Original Report: \n')
                original_report = tokenizer.decode_ids(val_actual_report[index].tolist())
                print(original_report)

                output_report = predictions[index,:]
                generated_report = tokenizer.decode_ids(output_report.tolist())
                print('\nPredicted Report: \n')
                print(generated_report)
        batch_val_loss.append(val_loss_components['report_loss'].to('cpu').item())
    training_loss.append(sum(batch_train_loss)/len(batch_train_loss))
    validation_loss.append(sum(batch_val_loss)/len(batch_val_loss))

    checkpoint_name = os.path.join(_C.CHECKPOINT_PATH, 'ck_pt_epoch_' + str(epoch) + '.pth')
    if(epoch%50==0 and epoch!=0):
        torch.save({'epoch': epoch,
                    'model_state_dict':model.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                    'train_loss':training_loss[-1],
                    'val_loss':validation_loss[-1]},
                   checkpoint_name)
        print('\nCheckpoint Saved for Epoch',epoch)
        print()
    print('\nTime Taken Per Epoch: ', time.time() - start_time)
    print('Training Loss for the Epoch ',epoch,'is: ',training_loss[-1])
    print('Validation Loss for the Epoch ', epoch, 'is: ', validation_loss[-1])
    print()
    torch.set_grad_enabled(True)
    model.train()
    scheduler.step()

print('Saving Model')

torch.save(model,_C.MODEL_PATH)
torch.save(model.state_dict(), _C.MODEL_STATE_DIC)
print("Model Saved")

util.create_csv_file(training_loss, validation_loss, os.path.join(_C.CSV_PATH, 'report_loss.csv'))

util.visualization(training_loss,validation_loss,'Report Loss','Epochs','Loss',
              'Report Train Loss', 'Report Val Loss', os.path.join(_C.FIGURE_PATH,'report_loss.png'))
print('\nTraining Took: ',time.time()-training_start_time)