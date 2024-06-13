import torch
import torch.nn as nn
import torch.nn.functional as F
import pefile
import argparse
import os
import pickle
from torch.utils.data import DataLoader, TensorDataset

# Định nghĩa lớp AttentionRCNN
class AttentionRCNN(nn.Module):
    def __init__(self, embed_dim, out_channels, window_size, module, hidden_size,
                 num_layers, bidirectional, attn_size, residual, dropout=0.5):
        super(AttentionRCNN, self).__init__()
        assert module.__name__ in {"RNN", "GRU", "LSTM"}, "`module` must be a `torch.nn` recurrent layer"
        self.residual = residual
        self.embed = nn.Embedding(257, embed_dim)
        self.conv = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=out_channels,
            kernel_size=window_size,
            stride=window_size,
        )
        self.rnn = module(
            input_size=out_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
        )
        rnn_out_size = (int(bidirectional) + 1) * hidden_size
        self.local2attn = nn.Linear(rnn_out_size, attn_size)
        self.global2attn = nn.Linear(rnn_out_size, attn_size, bias=False)
        self.attn_scale = nn.Parameter(
            nn.init.kaiming_uniform_(torch.empty(attn_size, 1))
        )
        self.dropout = nn.Dropout(dropout)
        if residual:
            self.fc = nn.Linear(out_channels + rnn_out_size, 1)
        else:
            self.fc = nn.Linear(rnn_out_size, 1)

    def forward(self, x):
        embedding = self.dropout(self.embed(x))
        conv_in = embedding.permute(0, 2, 1)
        conv_out = self.conv(conv_in)
        if self.residual:
            values, _ = conv_out.max(dim=-1)
        conv_out = conv_out.permute(2, 0, 1)
        rnn_out, _ = self.rnn(conv_out)
        global_rnn_out = rnn_out.mean(dim=0)
        attention = torch.tanh(
            self.local2attn(rnn_out) + self.global2attn(global_rnn_out)
        ).permute(1, 0, 2)
        alpha = F.softmax(attention.matmul(self.attn_scale), dim=-1)
        rnn_out = rnn_out.permute(1, 0, 2)
        fc_in = (alpha * rnn_out).sum(dim=1)
        if self.residual:
            fc_in = torch.cat((fc_in, values), dim=-1)
        output = self.fc(fc_in).squeeze(1)
        return output

def load_model(model_class, file_path, device='cpu', **model_params):
    model = model_class(**model_params)
    model.load_state_dict(torch.load(file_path, map_location=device))  # Tải trạng thái mô hình đã lưu
    model.to(device)
    model.eval()  # Chuyển mô hình sang chế độ đánh giá
    return model

def extract_features_from_pe(file_path, max_len=2048):
    try:
        file = pefile.PE(file_path)
        header = list(file.header)
        if len(header) > max_len:
            header = header[:max_len]
        else:
            header += [0] * (max_len - len(header))
        header_tensor = torch.tensor(header, dtype=torch.long)
        header_tensor = header_tensor.unsqueeze(0)
        return header_tensor
    except pefile.PEFormatError:
        print(f"Skipping {file_path}, not a valid PE file.")
        return None

def predict_pe_file(model, file_path, max_len, device='cpu'):
    input_tensor = extract_features_from_pe(file_path, max_len)
    if input_tensor is None:
        return None, None
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.sigmoid(output).item()
    label = 'malware' if prediction > 0.5 else 'benign'
    return prediction, label

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    model_params = {
        "embed_dim": 8,
        "out_channels": 128,
        "window_size": 32,
        "module": nn.GRU,
        "hidden_size": 128,  # Chỉnh sửa để khớp với mô hình đã lưu
        "num_layers": 2,  # Chỉnh sửa để khớp với mô hình đã lưu
        "bidirectional": True,  # Chỉnh sửa để khớp với mô hình đã lưu
        "attn_size": 64,  # Chỉnh sửa để khớp với mô hình đã lưu
        "residual": True,  # Chỉnh sửa để khớp với mô hình đã lưu
        "dropout": 0.5  # Chỉnh sửa để khớp với mô hình đã lưu
    }
    model = load_model(AttentionRCNN, args.model_path, device=args.device, **model_params)
    for file_name in os.listdir(args.input_dir):
        file_path = os.path.join(args.input_dir, file_name)
        prediction, label = predict_pe_file(model, file_path, args.max_len, device=args.device)
        if prediction is not None:
            print(f"File: {file_name}, Prediction: {prediction:.4f}, Label: {label}")
        else:
            print(f"Skipping {file_path}, not a valid PE file.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Directory of input PE files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output results.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model (.pt file).")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run the model on.")
    parser.add_argument("--max_len", type=int, default=2048, help="Maximum length of the input sequence.")
    args = parser.parse_args()
    
    main(args)

