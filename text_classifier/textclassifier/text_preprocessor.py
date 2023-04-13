from text_classifier.dataloader.preprocessor.sentence import run_prepro

from utils.opts import parse_opt

if __name__ == '__main__':
    config = parse_opt()

    if config.model_name in ['han']:
        raise NotImplementedError()
    else:
        run_prepro(
            csv_folder = config.dataset_path,
            output_folder = config.output_path,
            word_limit = config.word_limit,
            min_word_count = config.min_word_count
        )
