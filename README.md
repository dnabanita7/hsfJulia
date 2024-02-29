# hsfJulia

Follow these steps assuming you have Git and Julia (>1.10.1)installed:

1. **Clone the Repository**: Open your terminal or command prompt and run the following command to clone the `hsfJulia` repository:
   ```
   git clone https://github.com/dnabanita7/hsfJulia.git
   ```

2. **Navigate to the Repository**: Change into the `hsfJulia/src` directory:
   ```
   cd hsfJulia/src
   ```

3. **Open the Julia REPL**: Start the Julia REPL by simply typing `julia` in your terminal.

4. **Instantiate the Project**: Once in the Julia REPL, activate the project by running:
   ```julia
   ] activate .
   ```
   This will activate the `hsfJulia` project environment.

5. **Install Dependencies**: If the project has dependencies listed in the `Project.toml` file, you need to install them. This can be done by running:
   ```julia
   ] instantiate
   ```
   This command will install the necessary packages specified in the `Project.toml` file.

6. **Run the Script**: After activating the project and installing dependencies, you can run the `hsfJulia.jl` file. Exit the package manager mode by hitting the backspace key, and then run:
   ```julia
   include("hsfJulia.jl")
   ```
   This will execute the `hsfJulia.jl` script within the context of the project environment.

7. **Observe the Results**: The script should now execute, and you should see the desired results or any output generated by `hsfJulia.jl`.

By following these steps, you'll clone the `hsfJulia` repository, set up its Julia environment, and execute the main script within the context of that environment.


## EXPLANATION

In the provided code snippet, several choices regarding the neural network model, loss function, and optimizer have been made. Let's discuss why these choices were made:

    Model Architecture:
        The chosen model architecture consists of three fully connected (Dense) layers with 3, 64, and 32 units respectively, followed by a final output layer with 1 unit and a sigmoid activation function. This architecture is commonly used for binary classification tasks.
        The choice of a fully connected feedforward neural network is appropriate for this dataset as it allows the model to learn complex relationships between the input features and the binary classification labels.

    Loss Function:
        Binary cross-entropy loss (or log loss) is commonly used for binary classification problems where the output is a probability value between 0 and 1.
        The Flux.binarycrossentropy function is suitable for this task as it computes the binary cross-entropy loss between the predicted probabilities and the true binary labels.
        Binary cross-entropy loss is a natural choice here as it encourages the model to output probabilities close to 1 for positive samples and close to 0 for negative samples.

    Optimizer:
        The Adam optimizer is chosen for its effectiveness in training neural networks, especially in cases where the data is sparse or the parameters have different scales.
        The learning rate and other hyperparameters for the Adam optimizer are often chosen based on empirical performance and experimentation.
        The Flux.setup function is used to set up the optimizer for the given model. This function ensures that the optimizer is compatible with the model's parameters and updates them accordingly during training.

However, the chosen model architecture, loss function, and optimizer provide a solid foundation for the binary classification task, I remain open to exploring different configurations and fine-tuning the model to achieve even better results.

## TRAINING

Regarding the training process in the provided Julia code:

    Data Handling:
        The dataset is loaded from a CSV file using the CSV.read function and stored as a DataFrame.
        Features (input data) are extracted from the DataFrame and normalized using the mean and standard deviation.
        The dataset is split into training and validation sets. Half of the data is used for training, and the other half is used for validation.

    Model Training:
        The training loop runs for a specified number of epochs (10 in this case).
        For each epoch, the training data is passed through the model using Flux.train! function, which updates the model parameters using backpropagation and the specified optimizer.
        The loss function is calculated during training using the binary cross-entropy loss defined earlier.
        The training loop iterates over the entire training dataset in each epoch.

    Evaluation:
        After training, the model's performance is evaluated on the validation set.
        The validation data is passed through the trained model to obtain predictions.
        The accuracy is calculated by comparing the model's predictions to the true labels and averaging the results.

    GPU Acceleration:
        The code includes support for GPU acceleration using the CUDA and cuDNN packages, indicated by the lines using CUDA and using cuDNN.
        However, GPU usage is currently commented out (# Transfer data to GPU) in the provided code. Uncommenting these lines would enable GPU acceleration for data processing and model training.

Overall, the training process follows a standard supervised learning workflow:

    Data preprocessing (normalization, splitting).
    Model training using backpropagation and optimization.
    Evaluation of model performance on a separate validation set.

The code structure allows for easy modification and experimentation with different model architectures, loss functions, optimizers, and hyperparameters to improve model performance.
