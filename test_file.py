from dataloader import Create_DataLoader
import torch
import time
import os
import numpy as np
import pandas as pd
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
        test_dataloader = Create_DataLoader(_C.TEST_IMAGE_PATH, json_file_path=_C.TEST_JSON_PATH,
                                            max_sequence_length=_C.MAX_SEQUENCE_LENGTH, batch_size=_C.TEST_BATCH_SIZE,
                                            padding_idx=_C.PADDING_INDEX, tokeniser=tokenizer).test_dataloader_features_extracted()
    else:
        test_dataloader = Create_DataLoader(_C.TEST_IMAGE_PATH, json_file_path=_C.TEST_JSON_PATH,
                                            max_sequence_length=_C.MAX_SEQUENCE_LENGTH, batch_size=_C.TEST_BATCH_SIZE,
                                            padding_idx=_C.PADDING_INDEX, tokeniser=tokenizer).test_dataloader_images()

    torch.save(test_dataloader, _C.SAVED_DATASET_PATH_TEST)
else:
    test_dataloader = torch.load(_C.SAVED_DATASET_PATH_TEST)
visual_feature_size = 1536
torch.set_grad_enabled(False)
if(not _C.EXTRACTED_FEATURES):
    image_model = FeatureExtraction(torch.load(_C.IMAGE_MODEL_PATH))
    image_model.to(device)
    image_model.eval()

model = ReportGeneration(device=device, visual_feature_size=visual_feature_size, max_sequence_length=_C.MAX_SEQUENCE_LENGTH,
                         sos_index=_C.SOS_INDEX, eos_index=_C.EOS_INDEX, embedding_dim=_C.EMBEDDING_DIM, vocab_size=_C.VOCAB_SIZE,
                         num_layers=_C.COMBINED_N_LAYERS, attention_heads=_C.N_HEAD, drop_out=_C.DROPOUT_RATE,
                         padding_idx=_C.PADDING_INDEX, number_of_classes=_C.NUM_LABELS, beam_size=_C.BEAM_SIZE)

model.load_state_dict(torch.load(_C.MODEL_STATE_DIC))
model.to(device)
model.eval()

image_names = []
original_report = []
generated_report = []
for batch_test, data_test in enumerate(test_dataloader):
    if(batch_test>_C.ITERATIONS_PER_EPOCHS):
        break
    test_image_name, test_image, test_report = data_test
    
    if(not _C.EXTRACTED_FEATURES):
        test_image = image_model(test_image.to(device))
    test_image = test_image.to(device)
    test_report = test_report

    test_input = {'images':test_image, 'inference': _C.INFERENCE_TIME}

    test_output_dic = model(test_input)
    
    predictions = test_output_dic['predictions']
    for batch in range(predictions.shape[0]):
        output_report = predictions[batch,:]
        generated_report.append(tokenizer.decode(list(filter(lambda a: a != 0, output_report.tolist()))))
        original_report.append(tokenizer.decode(list(filter(lambda a: a != 0, test_report[batch,:].tolist()))))
        image_names.append(test_image_name[batch])
test_output = {'Image_Name':image_names, 'Original_Report': original_report, 'Generated_Report':generated_report}
test_df = pd.DataFrame.from_dict(test_output)
test_df.to_csv(_C.TEST_CSV_PATH, header=True, index=False)