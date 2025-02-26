# ggh

# AI-Powered RTL Depth Estimation using Random Forest

## Overview

This project implements a machine learning model to predict the combinational depth of signals in RTL designs. By using AI, we provide a faster alternative to synthesis-based timing analysis, enabling early detection of potential timing violations and reduced design cycle time. The core algorithm is a Random Forest Regressor trained on key RTL features.

## Setup

1.  **Prerequisites:**
    *   Python 3.8+
    *   pip
    *   [Icarus Verilog](http://iverilog.icarus.com/) (for generating depth reports - *optional, if you have another synthesis tool*)
2.  **Install Dependencies:**
    ```
    pip install pandas scikit-learn joblib
    ```
3.  **Environment Setup (Optional):**
    *   *If you have Icarus Verilog:* Ensure `iverilog` and `vvp` are in your system's PATH.

## Usage

1.  **Data Preparation:**
    *   **RTL Designs:** Place your Verilog RTL design files in the `data/rtl` directory.
    *   **Depth Reports:**
        *   *Option 1 (Icarus Verilog):* Use the provided `scripts/generate_depth_report.py` to automatically generate a CSV report from your RTL files. Edit the script to point to your RTL files, signal names, and desired clock period (for timing violation detection).  **See the "Generating Depth Reports" section below.**
        *   *Option 2 (Other Synthesis Tool):* Ensure you have a CSV file named `data/depth_report.csv` with columns: `rtl_file`, `signal_name`, `combinational_depth`. This CSV should contain the actual combinational depth values obtained from your synthesis tool. Make sure the `rtl_file` column has the relative path to the RTL file (e.g., `design/my_module.v`).
2.  **Model Training:**
    ```
    python src/train.py
    ```
    This script will train the Random Forest model using the data in `data/depth_report.csv` and save the trained model as `model/rtl_depth_predictor.joblib`.
3.  **Prediction:**
    ```
    python src/predict.py --rtl_file data/rtl/my_design.v --signal clk_out
    ```
    This command will use the trained model to predict the combinational depth of the signal `clk_out` in the RTL file `data/rtl/my_design.v`. The predicted depth will be printed to the console.

## Project Structure


## Algorithm Details

The core of the project is a Random Forest Regressor, implemented using scikit-learn, to predict combinational depth.

1.  **Feature Extraction:** The `feature_extraction.py` module parses the RTL code and extracts the following features for each signal:
    *   **Fan-in:** Number of inputs driving the signal's logic.
    *   **Fan-out:** Number of gate inputs the signal drives.
    *   **Signal Type:**  (e.g., register, net, input port).
    *   **Module Hierarchy Depth:**  Depth of the signal within the module instantiation tree.
2.  **Data Preprocessing:**
    *   Categorical features (Signal Type) are one-hot encoded.
    *   Numerical features (Fan-in, Fan-out, Module Hierarchy Depth) are standardized using `StandardScaler`.
3.  **Model Training:**  The `train.py` script loads the prepared data, trains the Random Forest model, and saves it for later use.
4.  **Prediction:**  The `predict.py` script loads the trained model, extracts features from a given RTL file and signal, and predicts the combinational depth.

## Generating Depth Reports (Optional - Icarus Verilog)

If you choose to use Icarus Verilog to generate your training data:

1.  **Install Icarus Verilog:** Follow the instructions on the [Icarus Verilog website](http://iverilog.icarus.com/).
2.  **Modify `scripts/generate_depth_report.py`:**
    *   Update the `rtl_directory` variable to point to your `data/rtl` directory.
    *   Modify the `signal_names` list to include the names of the signals you want to analyze.
    *   Adjust the `clock_period` variable to reflect the clock frequency of your design. This is used to identify potential timing violations.
3.  **Run the Script:**
    ```
    python scripts/generate_depth_report.py
    ```
    This will create or update the `data/depth_report.csv` file.
    **Note:** *The script assumes a simple design structure and might need adjustments for complex designs.*

## Evaluation

*   **Accuracy:** The model achieves a Mean Absolute Error (MAE) of on the test dataset.
*   **Runtime:** Prediction for a single signal takes approximately  seconds on average (measured on a \[Your CPU/System Specs]).

*   **Correctness and testing** Here are some of the tests we ran to ensure our model works correctly
    *   **Unit test**: Ran each function separately.
    *   **Sanity Check**: Compared with simulation output.
    *   **Edge Cases**: Ran and tested the model for edge cases.

## Team Members

*  Pranayani Kaushal

 
