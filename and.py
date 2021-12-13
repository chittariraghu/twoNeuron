from utils.model import Perceptron
from utils.all_utils import prepare_data, save_plot, save_model
import pandas as pd
import logging #Raghu : Added for logging

logging_str="[%(asctime)s: %(levelname)s: %(module)s] %(message)s"  #Raghu : Added for logging
logging.basicConfig(level=logging.INFO,format=logging_str)  #Raghu : Added for logging


def main(data, modelName, plotName, eta, epochs):
    df = pd.DataFrame(data)
    #Replace "print" with "logging.df" to use logging module
    #print(df)
    logging.info(f"This is actual dataframe{df}")
    X, y = prepare_data(df)
    model = Perceptron(eta=eta, epochs=epochs)
    model.fit(X, y)
    _ = model.total_loss()
    save_model(model, filename=modelName)
    save_plot(df, plotName, model)

if __name__ == '__main__':
    AND = {
        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y": [0,0,0,1],
    }
    ETA = 0.3 # 0 and 1
    EPOCHS = 10
    main(data=AND, modelName="and.model", plotName="and.png", eta=ETA, epochs=EPOCHS)