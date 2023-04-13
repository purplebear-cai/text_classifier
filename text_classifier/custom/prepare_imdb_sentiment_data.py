import pandas as pd


def convert_parquet_to_csv(in_path, out_path, split='train'):
      """
      Convert parquet file to csv file.
      """
      data = pd.read_parquet(in_path)
      texts = data['text'].values
      labels = data['label'].values

      labels_index = [label+1 for label in labels]
      converted_df = pd.DataFrame({'Label': labels_index,
                                   'Text': texts})
      converted_df.to_csv(out_path, columns=['Label', 'Text'])


train_in_path = '/Users/caiq/Workspace/adp/text-classifier/data/datasets/sentiment/aclImdb/grouped/train.parquet'
train_out_path = '/Users/caiq/Workspace/adp/text-classifier/data/datasets/sentiment/aclImdb/grouped/train.csv'
convert_parquet_to_csv(train_in_path, train_out_path, split='train')

test_in_path = '/Users/caiq/Workspace/adp/text-classifier/data/datasets/sentiment/aclImdb/grouped/test.parquet'
test_out_path = '/Users/caiq/Workspace/adp/text-classifier/data/datasets/sentiment/aclImdb/grouped/test.csv'
convert_parquet_to_csv(test_in_path, test_out_path, split='test')

# import glob
# import pandas as pd
# test_in_dir = '/Users/caiq/Workspace/adp/text-classifier/data/datasets/sentiment/aclImdb/test'
# test_pos = test_in_dir + '/pos'
# test_neg = test_in_dir + '/neg'
# labels = []
# texts = []
# for pos_file in glob.glob(test_pos + "/*.txt"):
#       with open(pos_file) as in_file:
#             lines = in_file.readlines()
#             for line in lines:
#                   labels.append(1)
#                   texts.append(line.strip())
#
# for neg_file in glob.glob(test_neg + "/*.txt"):
#       with open(neg_file) as in_file:
#             lines = in_file.readlines()
#             for line in lines:
#                   labels.append(0)
#                   texts.append(line.strip())
#
# df = pd.DataFrame({
#       'text': texts,
#       'label': labels
# })
# df.to_parquet('/Users/caiq/Workspace/adp/text-classifier/data/datasets/sentiment/aclImdb/grouped/test.parquet')

