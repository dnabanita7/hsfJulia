module hsfJulia

using CSV
using DataFrames
using Flux
using CUDA
using cuDNN
using Statistics
using Random
using LinearAlgebra

# Check if CUDA is available
if !CUDA.functional()
    error("CUDA is not available. Make sure you have a CUDA-compatible GPU and CUDA drivers installed.")
end

# Set device to GPU
device = gpu

# Load the CSV file
data = CSV.read("dataset.csv", DataFrame)

# Check the shape of the dataset
n_rows, n_cols = size(data)
println("Number of rows: ", n_rows)
println("Number of columns: ", n_cols)


features = Matrix(data[:, 1:3])  # Extract features (x, y, z)
features = Float32.(features)
labels = data[:, 4] .== "s"  # Convert labels to boolean (true for signal, false for background)

# Normalize features
μ = mean(features, dims=1)
σ = std(features, dims=1)
features = (features .- μ) ./ σ

# Transfer data to GPU
#features = device(features)
#labels = device(labels)

# Split data into training and validation sets
n = size(features, 1)
indices = shuffle(1:n)
train_indices = indices[1:div(n, 2)]
val_indices = indices[div(n, 2)+1:end]
train_data = features[train_indices, :], labels[train_indices]  # Pass features and labels
val_data = features[val_indices, :], labels[val_indices]  # Pass features and labels


println("data: ", size(train_data[1]))
println("labels: ", size(train_data[2]))
#transposed_data = transpose(train_data[1])
#println("training data: ", typeof(transposed_data))
println("data: ", typeof(train_data[1]))
println("labels: ", typeof(train_data[2]))

# Define neural network architecture
model = Chain(
    Dense(3, 64, relu),
    Dense(64, 32, relu),
    Dense(32, 1, sigmoid)
)

println("model architecture: ", model)
for layer in model.layers
    println(layer)
end


# Transfer model to GPU
#model = device(model)

# Define loss function
loss(model, x, y) = Flux.binarycrossentropy(model(x), y)

# Define optimizer
opt = Flux.setup(Adam(), model)

# Train the model
plain_data = copy(transpose(reshape(train_data[1], (50000, 3))))
plain_labels = collect(transpose(reshape(train_data[2], (50000,))))

final_data = [(plain_data[:, i], plain_labels[i]) for i in 1:size(plain_data, 2)]

epochs = 10
for epoch in 1:epochs
    println("Epoch ", epoch, " starts...")
    Flux.train!(loss, model, final_data, opt)
end

# Evaluate model on validation set
plain_val_data = copy(transpose(reshape(val_data[1], (50000, 3))))
plain_val_labels = collect(transpose(reshape(val_data[2], (50000,))))

accuracy = mean(Flux.isequal.(model(plain_val_data) .> 0.5, plain_val_labels))
println("Validation accuracy: $accuracy") # Validation accuracy: 0.9831






end # module hsfJulia
