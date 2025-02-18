from src.pyclassify.classifier import kNN
from src.pyclassify.utils import read_config, read_file
import os
import argparse

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', default='config/config')
    args = argparser.parse_args()

    # i. Read configuration
    config = read_config(args.config)
    
    # Read dataset
    features, labels = read_file(config['dataset'])
    
    # ii. Split data into train and test
    split_point = int(len(features) * 0.2)
    X_train = features[:split_point]
    y_train = labels[:split_point]
    X_test = features[split_point:]
    y_test = labels[split_point:]
    
    # iii. Perform kNN classification
    classifier = kNN(k=config['k'])
    predictions = classifier((X_train, y_train), X_test)
    
    # Calculate accuracy
    correct = sum(1 for pred, true in zip(predictions, y_test) if pred == true)
    accuracy = correct / len(y_test)
    print(f"Classification accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main()