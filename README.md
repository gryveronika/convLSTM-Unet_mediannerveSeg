### convLSTM+U-NET: How to run the code

Th Github repository does not provide data necessary to train the segmentation, due to GDPR.
Therefore, the folders where to save results, are empty. 
Fill out the filepaths to your data, YOUR_DATA_PATH in the config file.


#### 1. Preprocess data, create datasets and dataloaders

   - **File:** `dataprocessing.py`
   - **Purpose:** This script preprocesses the data, creates a dataset which combines an image sequence with the 
ground truth belonging to the last image in the sequence, and make dataloaders. 
   - **Command:** Execute the following command:
     ```bash
     python dataprocessing.py

#### 2. Training the Model

   - **File:** `main.py`
   - **Purpose:** This script is responsible for training the model.
   - **Command:** Execute the following command:
     ```bash
     python main.py
     ```

#### 3. Evaluating the Model

   - **File:** `evaluate.py`
   - **Purpose:** This script saves prediction data for the test set.
   - **Command:** Run the following command:
     ```bash
     python evaluate.py
     ```

#### 4. Calculating Metrics

   - **File:** `get_metrics.py`
   - **Purpose:** This script calculates accuracy, precision, recall, F1 score/Dice, and Intersection over Union (IoU) for all classes seperately.
   - **Command:** Execute the following command:
     ```bash
     python get_metrics.py
     ```

#### 5. Plotting segmentation

   - **File:** `plot_segmentation.py`
   - **Purpose:** This script generates plots based on the predictions saved in `evaluate.py`.
   - **Command:** Run the following command:
     ```bash
     python plot_result.py
     ```
### Config and folders 

#### Config

   - **File:** `config.py`
   - **Purpose:** This file includes definitions of number of classes, number of unet-levels etc. This file is usually not ran, but imported in other files.

#### Source code folder

- **Purpose:** The folder includes a training function + class definitions such as convLSTM+U-Net and DiceLoss. 

#### Plot code folder

   - **Purpose:** This folder includes some of the plotting scripts used to generate the plots in the report.
