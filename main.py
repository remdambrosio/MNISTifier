# Title: main.py
# Authors: Rem D'Ambrosio
# Created: 2024-12-02
# Description: initial structure by Mayur Ingole

from Trainer import Trainer  
from Tester import Tester
import argparse


def main():
    parser = argparse.ArgumentParser(description='training and testing neural network')
    parser.add_argument('-tr', '--train', action='store_true', help='training cnn')
    parser.add_argument('-te', '--test', action='store_true', help='testing cnn')
    parser.add_argument('-de', '--demo', action='store_true', help='demoing cnn on sample')
    parser.add_argument('-mo', '--model', type=str, help='model name', default='model1')
    parser.add_argument('-sa', '--sample', type=str, help='sample name', default='sample1')
    args = parser.parse_args()

    model_path = 'models/' + args.model + '.pth'
    sample_path = 'samples/' + args.sample + '.bmp'

    if args.train:     
        train(model_path)

    if args.test:
        test(model_path)

    if args.demo:
        demo(model_path, sample_path)

    return
  

def train(model_path):
    trainer = Trainer()
    model = trainer.train_model(10)
    trainer.save_model(model, model_path)
    return


def test(model_path):
    tester = Tester()
    model = tester.load_model(model_path)
    tester.test_model(model)
    return


def demo(model_path, sample_path):
    tester = Tester()
    model = tester.load_model(model_path)
    tester.demo_model(model, sample_path)
    return


if __name__ == '__main__':
    main()