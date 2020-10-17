import json
import os
import tempfile
import unicodedata
from typing import List
import pandas as pd
from tqdm import tqdm
import sentencepiece as sp

vocab_size = 2100
vocab_output_path = '/netscratch/gsingh/MIMIC_CXR/DataSet/Indiana_Chest_XRay/Vocab/indiana'
# input_report_csv_path = '/netscratch/gsingh/MIMIC_CXR/DataSet/MIMIC_CXR_Reports/Report_CSV_Files/cxr-study-list.csv'
# init_report_path = '/netscratch/gsingh/MIMIC_CXR/DataSet/MIMIC_CXR_Reports/Complete_reports'
#
# report_path_df = pd.read_csv(input_report_csv_path)
# report_path_list = list(report_path_df['path'])
# reports = list()
#
# for index, report_value in tqdm(enumerate(report_path_list)):
#     report_path = os.path.join(init_report_path,report_value)
#     with open(report_path, 'r') as file:
#         report = file.read()
#     report = report.lower()
#     # report = unicodedata.normalize('NFKD', report)
#     # report = "".join([chr for chr in report if not unicodedata.combining(chr)])
#     reports.append(report)
complete_txt_path = '/netscratch/gsingh/MIMIC_CXR/DataSet/Indiana_Chest_XRay/iu_xray_combined_reports_filtered_stop_words_not_removed.txt'
# with open(complete_txt_path, "w") as reports_file:
#     for report in tqdm(reports):
#         reports_file.write(report + "\n")

# train sentencepiece model from `botchan.txt` and makes `m.model` and `m.vocab`
# `m.vocab` is just a reference. not used in the segmentation.
sp.SentencePieceTrainer.train('--input=/netscratch/gsingh/MIMIC_CXR/DataSet/Indiana_Chest_XRay/iu_xray_combined_reports_filtered_stop_words_not_removed.txt'
                               ' --pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 --pad_piece=[PAD] --unk_piece=[UNK] --bos_piece=[BOS] --eos_piece=[EOS]'
                               ' --model_prefix=/netscratch/gsingh/MIMIC_CXR/DataSet/Indiana_Chest_XRay/Vocab/indiana'
                               ' --model_type=bpe'
                               ' --vocab_size=3000')

# makes segmenter instance and loads the model file (m.model)
aa = sp.SentencePieceProcessor()
aa.load('/netscratch/gsingh/MIMIC_CXR/DataSet/Indiana_Chest_XRay/Vocab/indiana.model')
#
# # encode: text => id
# print(sp.encode_as_ids('lungs <unk>'))
# print(sp.decode_ids([113, 2073, 1, 40, 2097, 1]))
# print('AAA')
# x = [2,113]
# for i in range (10):
#     x.append(0)
# x.append(3)
# x.append(1)
# print((x))
# print(sp.decode_ids(x))
# # print(sp.decode_ids([1]))
# # print(sp.decode_ids([0]))
# # print(sp.decode_ids([2]))
# # # decode: id => text
# print('AAAA')
print('bos=', aa.bos_id())
print('eos=', aa.eos_id())
print('unk=', aa.unk_id())
print('pad=', aa.pad_id())
# #
# sp_user = sp.SentencePieceProcessor()
# sp_user.load('/netscratch/gsingh/MIMIC_CXR/DataSet/Indiana_Chest_XRay/Vocab/indiana.model')
#
# encoded = sp_user.encode('<s>findings  the </s>')
# decoded = sp_user.decode(encoded)
# print('Encoded: ',encoded)
# print('Length Encoded: ', len(encoded))
# print('Decoded: ',decoded)
# print(sp_user.decode([2060, 2093]))
#
# print(sp_user.decode([2060, 0,0, 0, 2093]))
# print('bos=', sp.bos_id())
# print('eos=', sp.eos_id())
# print('unk=', sp.unk_id())
# print('pad=', sp.pad_id())