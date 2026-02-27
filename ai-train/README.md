pip install torch torchvision tqdm

dataset/
 ├── train/
 │    ├── class0/
 │    ├── class1/
 │    └── class2/
 └── val/
      ├── class0/
      ├── class1/
      └── class2/


# offline train
local_model/
 ├── config.json
 ├── pytorch_model.bin
 ├── tokenizer.json
 ├── tokenizer_config.json
 └── vocab.txt


 pip download torch torchvision transformers -d offline_packages

 pip install --no-index --find-links=offline_packages torch torchvision transformers

# Force Transformers to Work Offline

# VERY IMPORTANT.

# In your training script:

    import os
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"