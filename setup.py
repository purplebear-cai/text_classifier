import setuptools

setuptools.setup(
    name="text_classifier",
    version="0.0.5",
    author="Qingqing Cai",
    description="Generic Text Classifier",
    packages=['text_classifier/custom', 'text_classifier/textclassifier',
              'text_classifier/dataloader', 'text_classifier/eda',
              'text_classifier/models', 'text_classifier/trainer',
              'text_classifier/utils', 'text_classifier/dataloader/preprocessor',
              'text_classifier/models/MLP', 'text_classifier/models/SentEncoderMLP',
              'text_classifier/models/TextCNN', 'text_classifier/models/Transformer'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    py_modules=["text_classifier"],
    # package_dir={'':'text_classifier'},
    install_requires=[]
)
