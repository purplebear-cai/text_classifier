from eda.data_explorer import get_num_words_per_sample


def get_ratio_of_samples_over_words_counts(sample_texts):
    """
    Get the number of samples / number of words per sample ration.
    If this ratio is < 1500, tokenize the text as n-grams and use MLP to classify them.
    If this ratio is > 1500, tokenize the text as sequence and use sepCNN model to classifier them.

    # Arguments
        sample_texts: list, sample texts.
    # Returns
        float, ratio of number of samples / number of words per sample.
    """
    number_of_samples = len(sample_texts)
    num_words_per_sample = get_num_words_per_sample(sample_texts)
    return number_of_samples * 1.0 / num_words_per_sample * 1.0