# KLTN
Dikedataset: https://github.com/iosifache/DikeDataset.git

Cấu hình của model: model.py

Cài đặt train, test: utils.py

Tiến hành train: python3.6 train.py --benign_dir path/to/benign --malware_dir path/to/malware --model RCNN --rnn_module LSTM

Load model: python3.6 loadAttnRCNN.py --input_dir path_to_pe_files --output_dir path_to_save_results --model_path path_to_saved_model.pt 

python3.6 loadMalConvPlus.py --input_dir path_to_pe_files --output_dir path_to_save_results --model_path path_to_saved_model.pt 
