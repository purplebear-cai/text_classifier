import pandas as pd


def convert_parquet_to_csv(in_path, out_path, split='train', label_map=None):
      """
      Convert parquet file to csv file.
      """
      data = pd.read_parquet(in_path)
      texts = data['sentence'].values
      labels = data['label'].values
      if split == 'train':
        labels_set = list(set(labels))
        label_map = {label: idx+1 for idx, label in enumerate(labels_set)}
        rev_label_map = {idx: label for label, idx in label_map.items()}
      labels_index = [label_map[label] for label in labels]
      converted_df = pd.DataFrame({'Label': labels_index,
                                   'Text': texts})
      converted_df.to_csv(out_path, columns=['Label', 'Text'])
      if split == 'train':
        return label_map, rev_label_map
      else:
        return None

train_in_path = '/Users/caiq/Workspace/adp/text-classifier/data/datasets/cr_sents/multi_sents.train.parquet'
train_out_path = '/Users/caiq/Workspace/adp/text-classifier/data/datasets/cr_sents/train.csv'
label_map, _ = convert_parquet_to_csv(train_in_path, train_out_path, split='train')

test_in_path = '/Users/caiq/Workspace/adp/text-classifier/data/datasets/cr_sents/multi_sents.test.parquet'
test_out_path = '/Users/caiq/Workspace/adp/text-classifier/data/datasets/cr_sents/test.csv'
convert_parquet_to_csv(test_in_path, test_out_path, split='test', label_map=label_map)