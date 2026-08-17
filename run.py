import sys
import os
import argparse
from src.evaluate import evaluate

class DummyArgs:
    def __init__(self, input_dir, output_dir, model_path):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.model_path = model_path
        self.gt_dir = None
        self.base_ch = 32

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python run.py <input-dir> <output-dir>")
        sys.exit(1)
        
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    model_path = os.path.join("models", "best_model.pth")
    
    args = DummyArgs(input_dir, output_dir, model_path)
    evaluate(args)
