# Blurred Encoding for Trajectory Representation Learning

Since the trajectory data is too large, we only provide sample data in `./data`, and if the paper is accepted, we will open source code and data.


### Code directory

```
BLUE/
├── config/                  # Parameter
│   ├── chengdu.yaml
│   └── porto.yaml
├── data/                    # dataset
├── datasets.py              # Load dataset
├── downstream/              # Downstream tasks execution
│   ├── classification.py
│   ├── downstream_utils.py
│   ├── similarity.py
│   └── travel_time.py
├── downstream_main.py       # Downstream tasks main and config
├── logs/                    # log file
├── model/                   # model architecture
│   ├── embedding.py
│   └── network.py
├── preprocess.py            # dataset preproces
├── time_features.py         # get time feature
├── train.py                 # pre-train
└── utils.py                 # some utility functions
```

## Training
```
# Chengdu
python train.py --config chengdu --exp_id <SET_YOUR_ID> --device cuda:0

# Porto
python train.py --config porto --exp_id <SET_YOUR_ID> --device cuda:0
```

## Downstream tasks
When run the model for the downstream tasks, set the same **exp_id** in `train.py` as for pre-training.

### Travel Time Estimation (Fine-tuning)
```
# Chengdu
python downstream_main.py --config chengdu --exp_id <SET_YOUR_ID> --task travel_time --device cuda:0 

# Porto
python downstream_main.py --config porto --exp_id <SET_YOUR_ID> --task travel_time --device cuda:0 
```

### Trajectory classification (Fine-tuning)
```
# Chengdu
python downstream_main.py --config chengdu --exp_id <SET_YOUR_ID> --task classification --device cuda:0 

# Porto
python downstream_main.py --config porto --exp_id <SET_YOUR_ID> --task classification --device cuda:0 
```

### Most similar trajectory search (No Fine-tuning)
```
# Chengdu
python downstream_main.py --config chengdu --exp_id <SET_YOUR_ID> --task similarity --device cuda:0 

# Porto
python downstream_main.py --config porto --exp_id <SET_YOUR_ID> --task similarity --device cuda:0 
```


## Transfer tasks
When run the model for the transfer tasks, set the same **exp_id** in `train.py` as for pre-training.

### Travel Time Estimation (Fine-tuning)
```
# Chengdu->porto
python downstream_main.py --config chengdu --exp_id <SET_YOUR_ID> --task travel_time --device cuda:0 --transfer True --transfer_config porto

# Porto->chengdu
python downstream_main.py --config porto --exp_id <SET_YOUR_ID> --task travel_time --device cuda:0 --transfer True --transfer_config chengdu
```

### Trajectory classification (Fine-tuning)
```
# Chengdu->porto
python downstream_main.py --config chengdu --exp_id <SET_YOUR_ID> --task classification --device cuda:0 --transfer True --transfer_config porto

# Porto->chengdu
python downstream_main.py --config porto --exp_id <SET_YOUR_ID> --task classification --device cuda:0 --transfer True --transfer_config chengdu
```

### Most similar trajectory search (No Fine-tuning)
```
# Chengdu->porto
python downstream_main.py --config chengdu --exp_id <SET_YOUR_ID> --task similarity --device cuda:0 --transfer True --transfer_config porto

# Porto->chengdu
python downstream_main.py --config porto --exp_id <SET_YOUR_ID> --task similarity --device cuda:0 --transfer True --transfer_config chengdu
```
