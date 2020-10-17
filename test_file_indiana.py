from dataloader import Create_DataLoader
import torch
import time
import os
import numpy as np
# print('Inside Test file')
from report_gen_indiana import ReportGeneration

from config_indiana import Config
import utilities as util
import sentencepiece as sp
import warnings
import pandas as pd


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_C = Config()
image_model, visual_feature_size, image_size = _C.MODELS[0][_C.MODEL_NAME]

tokenizer = sp.SentencePieceProcessor()
tokenizer.load(_C.VOCAB_MODEL_PATH)


if(not _C.SAVED_DATASET):
    test_dataloader = Create_DataLoader(image_path=_C.IMAGE_PATH, json_file_path=_C.TEST_JSON_PATH, shuffle=True,
                                            max_sequence_length=_C.MAX_SEQUENCE_LENGTH, batch_size=1, image_size=image_size, sos_idx=_C.SOS_INDEX, eos_idx=_C.EOS_INDEX,
                                            padding_idx=_C.PADDING_INDEX, tokeniser=tokenizer, random_state=_C.RANDOM_SEED).create_indiana_dataset()

    torch.save(test_dataloader, _C.SAVED_DATASET_PATH_TEST)
else:
    test_dataloader = torch.load(_C.SAVED_DATASET_PATH_TEST)

torch.set_grad_enabled(False)
# if(not _C.EXTRACTED_FEATURES):
#     image_model = FeatureExtraction(torch.load(_C.IMAGE_MODEL_PATH))
#     image_model.to(device)
#     image_model.eval()


model = ReportGeneration(device=device, image_model=image_model, visual_feature_size=visual_feature_size, max_sequence_length=_C.MAX_SEQUENCE_LENGTH,
                         sos_index=_C.SOS_INDEX, eos_index=_C.EOS_INDEX, embedding_dim=_C.EMBEDDING_DIM, vocab_size=_C.VOCAB_SIZE,
                         num_layers=_C.COMBINED_N_LAYERS, attention_heads=int(_C.EMBEDDING_DIM/_C.D_HEAD), drop_out=_C.DROPOUT_RATE,
                         padding_idx=_C.PADDING_INDEX, number_of_classes=_C.NUM_LABELS, use_beam_search=_C.USE_BEAM_SEARCH, beam_size=_C.BEAM_SIZE)

model.load_state_dict(torch.load(_C.MODEL_STATE_DIC))
model.to(device)
model.eval()

image_names = []
original_report = []
generated_report = []
print('Length of test dataloader: ',len(test_dataloader))
from tqdm import tqdm
for batch_test, data_test in enumerate(test_dataloader):
    # if(batch_test>_C.ITERATIONS_PER_EPOCHS):
    #     break
    test_image_name, test_image, test_input_report, test_actual_report = data_test

    # if(not _C.EXTRACTED_FEATURES):
    #     test_image = image_model(test_image.to(device))
    test_image = test_image.to(device)
    test_report = test_actual_report

    test_input = {'images':test_image, 'inference': _C.INFERENCE_TIME}

    test_output_dic = model(test_input)
    predictions = test_output_dic['predictions']
    image_names.append(test_image_name)
    generated_report.append(tokenizer.decode_ids(predictions[0].tolist()))
    original_report.append(tokenizer.decode_ids(test_report[0].tolist()))


#     for batch in range(predictions.shape[0]):
#         output_report = predictions[batch,:]
#         generated_report.append(tokenizer.decode(list(filter(lambda a: a != 0, output_report.tolist()))))
#         original_report.append(tokenizer.decode(list(filter(lambda a: a != 0, test_report[batch,:].tolist()))))
#         image_names.append(test_image_name[batch])
test_output = {'Image_Name':image_names,'Generated_Report':generated_report, 'Original_Report':original_report}
test_df = pd.DataFrame.from_dict(test_output)
test_df.to_json(_C.TEST_CSV_PATH, orient='index')