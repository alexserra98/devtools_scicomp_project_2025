import argparse
from pyclassify import kNN
from pyclassify.utils import read_config, read_file

def main(config_file):
    # Obtain kwargs using read_config
    kwargs = read_config(config_file)
    
    # Check if 'k' is in the config, raise an error if not
    if 'k' not in kwargs:
        raise KeyError("The configuration file must contain the 'k' parameter.")
    if 'dataset' not in kwargs:
        raise KeyError("The configuration file must contain the 'dataset' parameter.")
    
    # Download the data
    X, y = read_file(kwargs['dataset'])
    split = int(0.2 * len(X))
    X_train = X[:split]
    X_test = X[split:]
    y_train = y[:split]
    y_test = y[split:]

    # Create kNN instance with the value of k from the config
    knn_model = kNN(k=kwargs['k'])

    # Fit the model and compute accuracy
    y_hat = knn_model(data=(X_train, y_train), new_points=X_test)
    accuracy = sum(1 for true, pred in zip(y_test, y_hat) if true == pred) / len(y_test)
    print(f'Accuracy {accuracy:.2%}')

if __name__ == '__main__':
    # Set up argument parser to take the config file name as input
    parser = argparse.ArgumentParser(description="Run kNN model with a given config file.")
    parser.add_argument('--config_file', type=str, help="The name of the YAML config file (without .yaml)")
    
    # Parse the command line arguments
    args = parser.parse_args()
    
    # Call the main function with the provided config file
    try:
        main(args.config_file)
    except KeyError as e:
        print(f"Error: {e}")
        exit(1)  # Exit the program with a non-zero status indicating failure
