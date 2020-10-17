import matplotlib.pyplot as plt
import pandas as pd
import os

def visualization(first, second, title, x_label, y_label, legend_first, legend_second, save_path):
  plt.figure()
  plt.plot(first, label=legend_first)
  plt.plot(second, label=legend_second)
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.title(title)
  plt.legend()
  plt.savefig(save_path, dpi=350)
  plt.close()

def create_csv_file(list_1, list_2, csv_path):
  df = pd.DataFrame({'Train Loss':list_1, 'Validation Loss':list_2})
  df.to_csv(csv_path, index=False, header=True)


def output_results(image_names, gen_labels,labels, gen_report, real_report, csv_path):
  output_dic = {'image_name': [],  'Generated_Report': [], 'Actual_Report': []}

  if not os.path.exists(csv_path):
    df = pd.DataFrame.from_dict(output_dic)
    df.to_csv(csv_path,index=False)

  for index, inputs in enumerate(image_names):
    output_dic['image_name'].append(inputs.split('/')[-1])

  for index, report in enumerate(gen_report):
    output_dic['Generated_Report'] = report

  for index, report in enumerate(real_report):
    output_dic['Actual_Report'] = report


  input_df = pd.DataFrame.from_dict(output_dic)
  input_df.to_csv(csv_path, mode='a',header=False, index=False)

