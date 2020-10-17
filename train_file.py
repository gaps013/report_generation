from dataloader import Create_DataLoader
import torch
import time
import os
import numpy as np
from report_gen import ReportGeneration
from config import Config
import utilities as util
from visual_model import FeatureExtraction
from sentence_piece_tokenizer import SentencePieceBPETokenizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_C = Config()
tokenizer = SentencePieceBPETokenizer(_C.VOCAB_FILE_PATH, _C.VOCAB_MODEL_PATH)
if(not _C.SAVED_DATASET):
    if(_C.EXTRACTED_FEATURES):
        train_dataloader = Create_DataLoader(_C.TRAIN_IMAGE_PATH, json_file_path=_C.TRAIN_JSON_PATH, shuffle=True,
                                             max_sequence_length=_C.MAX_SEQUENCE_LENGTH, batch_size=_C.BATCH_SIZE,
                                             padding_idx=_C.PADDING_INDEX, tokeniser=tokenizer).train_val_dataloader_features_extracted(train=True,
                                                                                                          size=_C.TRAIN_DATASET_LENGTH)
        val_dataloader = Create_DataLoader(_C.VALID_IMAGE_PATH, json_file_path=_C.VALID_JSON_PATH,
                                           batch_size=_C.BATCH_SIZE, max_sequence_length=_C.MAX_SEQUENCE_LENGTH,
                                           padding_idx=_C.PADDING_INDEX, tokeniser=tokenizer).train_val_dataloader_features_extracted(train=False)


    else:
        train_dataloader = Create_DataLoader(_C.TRAIN_IMAGE_PATH, json_file_path=_C.TRAIN_JSON_PATH, shuffle=True,
                                             max_sequence_length=_C.MAX_SEQUENCE_LENGTH, batch_size=_C.BATCH_SIZE,
                                             padding_idx=_C.PADDING_INDEX, tokeniser=tokenizer).train_val_dataloader_images(train=True,
                                                                                                          size=_C.TRAIN_DATASET_LENGTH)
        val_dataloader = Create_DataLoader(_C.VALID_IMAGE_PATH, json_file_path=_C.VALID_JSON_PATH,
                                           batch_size=_C.BATCH_SIZE, max_sequence_length=_C.MAX_SEQUENCE_LENGTH,
                                           padding_idx=_C.PADDING_INDEX, tokeniser=tokenizer).train_val_dataloader_images(train=False)

    torch.save(train_dataloader, _C.SAVED_DATASET_PATH_TRAIN)
    torch.save(val_dataloader, _C.SAVED_DATASET_PATH_VAL)
else:
    train_dataloader = torch.load(_C.SAVED_DATASET_PATH_TRAIN)
    val_dataloader =  torch.load(_C.SAVED_DATASET_PATH_VAL)

training_loss = []
validation_loss = []
if(not _C.EXTRACTED_FEATURES):
    image_model = FeatureExtraction(torch.load(_C.IMAGE_MODEL_PATH))
    image_model.to(device)
    image_model.eval()
visual_feature_size = 1536

model = ReportGeneration(device=device, visual_feature_size=visual_feature_size, max_sequence_length=_C.MAX_SEQUENCE_LENGTH,
                         sos_index=_C.SOS_INDEX, eos_index=_C.EOS_INDEX, embedding_dim=_C.EMBEDDING_DIM, vocab_size=_C.VOCAB_SIZE,
                         num_layers=_C.COMBINED_N_LAYERS, attention_heads=_C.N_HEAD, drop_out=_C.DROPOUT_RATE,
                         padding_idx=_C.PADDING_INDEX, number_of_classes=_C.NUM_LABELS, beam_size=_C.BEAM_SIZE)
model.to(device)
torch.set_grad_enabled(True)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=_C.LR_COMBINED, weight_decay=_C.WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=_C.MAX_LR, epochs=_C.EPOCHS, steps_per_epoch=len(train_dataloader))

print('Going inside tokens')
for epoch in range(_C.EPOCHS):
    start_time = time.time()
    batch_train_loss = []
    batch_val_loss = []
    for batch_train, data_train in enumerate(train_dataloader):
        train_image_name, train_image, train_report = data_train
        report_length = torch.Tensor(np.full((train_report.size(0),),_C.MAX_SEQUENCE_LENGTH))
        if(not _C.EXTRACTED_FEATURES):
            train_image = image_model(train_image.to(device))
        train_image = train_image.to(device)
        train_report = train_report.to(device)
        report_length = report_length.to(device)
        optimizer.zero_grad()
        input = {'images':train_image, 'inference': _C.INFERENCE_TIME,
                 'tokenised_report':train_report, 'report_length':report_length}
        output_dic = model(input)
        train_loss, train_loss_components = output_dic['loss'], output_dic['loss_components']
        # print(type(train_loss), type(train_loss_components))
        batch_train_loss.append(train_loss_components['report_loss'].to('cpu').item())
        train_loss.backward()
        optimizer.step()
        scheduler.step()
        if(batch_train%100==0 and batch_train!=0):
            print('Currently training Batch ',batch_train,' for epoch', epoch, ' and losses for current batch are ', batch_train_loss[-1])
            print()
    torch.set_grad_enabled(False)
    model.eval()
    print('Inside validation')
    for batch_val, data_val in enumerate(val_dataloader):
        val_image_name, val_image, val_report = data_val
        val_report_length = torch.Tensor(np.full((val_report.size(0),),_C.MAX_SEQUENCE_LENGTH))
        if(not _C.EXTRACTED_FEATURES):
            val_image = image_model(val_image.to(device))
        val_image = val_image.to(device)
        val_report = val_report.to(device)
        val_report_length = val_report_length.to(device)
        intersection_list = list(set(val_image_name) & set(_C.VALID_IMAGES))

        val_input = {'images':val_image, 'inference': _C.INFERENCE_TIME, 'tokenised_report':val_report,
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
            index = val_image_name.index(intersection_list[0])
            print('Val image name: ',intersection_list[0])
            print('\n Original Report: \n')
            original_report = tokenizer.decode(list(filter(lambda a: a != 0, val_report[index].tolist())))
            print(original_report)

            output_report = predictions[index,:]
            generated_report = tokenizer.decode(list(filter(lambda a: a != 0, output_report.tolist())))
            print('Predicted Report: \n')
            print(generated_report)
        batch_val_loss.append(val_loss_components['report_loss'].to('cpu').item())
    torch.set_grad_enabled(True)
    model.train()

    training_loss.append(sum(batch_train_loss)/len(batch_train_loss))
    validation_loss.append(sum(batch_val_loss)/len(batch_val_loss))

    checkpoint_name = os.path.join(_C.CHECKPOINT_PATH, 'ck_pt_epoch_' + str(epoch) + '.pth')
    if(epoch%5==0):
        torch.save({'epoch': epoch,
                    'model_state_dict':model.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                    'train_loss':training_loss[-1],
                    'val_loss':validation_loss[-1]},
                   checkpoint_name)
        print('Checkpoint Saved for Epoch',epoch)
        print()
    print('Time Taken Per Epoch: ', time.time() - start_time)
    print('Training Loss for the Epoch ',epoch,'is: ',training_loss[-1])
    print('Validation Loss for the Epoch ', epoch, 'is: ', validation_loss[-1])
    print()

print('Saving Model')

torch.save(model,_C.MODEL_PATH)
torch.save(model.state_dict(), _C.MODEL_STATE_DIC)
print("Model Saved")

util.create_csv_file(training_loss, validation_loss, os.path.join(_C.CSV_PATH, 'report_loss.csv'))

util.visualization(training_loss,validation_loss,'Report Loss','Epochs','Loss',
              'Report Train Loss', 'Report Val Loss', os.path.join(_C.FIGURE_PATH,'report_loss.png'))
