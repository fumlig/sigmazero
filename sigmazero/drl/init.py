# compile model with its initial state
# save as .pt file
from model import SigmaModel
import torch
from pathlib import Path
import os

def main(DIRECTORY='data', FILENAME='complete_model.pt'):
    model = SigmaModel()
    Path(DIRECTORY).mkdir(parents=True, exist_ok=True)
    torch.save(model, os.path.join(DIRECTORY, FILENAME))

if __name__ == '__main__':
    main()