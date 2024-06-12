# birdclef-2024

Solution to Birdclef 2024 challenge on Kaggle, CPMP part. The full solution is described in Kaggle forums at .

My code was all in notebooks given the data and the models I used were small and did not require any distributed training.

The code was run with nivida NGC container `nvidia/pytorch:24.02-py3`

I pip installed torchaudio, torchvision and timm packages in that container, then ran the notebooks.

The data needs to be downloaded from Kaggle. It includes data from these competitions and datasets:

- https://www.kaggle.com/competitions/birdclef-2024
- https://www.kaggle.com/competitions/birdclef-2023
- https://www.kaggle.com/competitions/birdclef-2022
- https://www.kaggle.com/competitions/birdclef-2021
- https://www.kaggle.com/competitions/birdsong-recognition
- https://www.kaggle.com/datasets/ludovick/birdclef2024-additional-mp3

The notebooks assume that the downloaded data is put in subdirecties of `/cpmp/` . The paths to data is defined in the cell at the top of each notebooks for an easy adaptation to any specific local setting.

The notebookd 002 processes this year competition data. For the trianing data it saves numpy versions of 10 second clips from the start and the end of each record. It also saves each sourdscape data as a numpy array.

The notebook 210 processed past competitions data and addiitnal data from Xeno Canto. It only saves data from the same species as this year competition. It also only saves data for species having less than 500 samples in the compeititon data. 

The notebook 234 trains 5 first level efficientvit_b0 models then predicts pseudo labels on unlabelled soundscapes. These are saved together with mdoel checkpoints in a directory printed near the start of the notebook. That directory needs to be cut and past into the next notebook.

The notebook 237 trains 5 second level efficientvit_b0 models. pseudo labels for unlabelled s9undscapes are read from the directory populated by the previous notebook. That directory must be set as the value of `cfg.pl` in the first cell.
