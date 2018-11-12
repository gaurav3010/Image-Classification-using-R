#load packages
library(EBImage)
library(keras)
library(tensorflow)

path = 'C:/Users/Rony/Desktop/DA Project'
save_in0 <- "C:/Users/Rony/Desktop/DA Project/final_dataset/n0"
save_in1 <- "C:/Users/Rony/Desktop/DA Project/final_dataset/n1"

#------------------------- n0 ---------------------------------#
#Read Images
setwd('C:/Users/Rony/Desktop/DA Project/Dataset/n0')
pics <- list.files(path = ".")

for(i in 1:length(pics))
{
  # Try-catch is necessary since some images
  # may not work.
  result <- tryCatch({
    # Image name
    imgname <- pics[i]
    # Read image
    img <- readImage(imgname)
    # Resize image 28x28
    img_resized <- resize(img, w = 28, h = 28)
    # Set to grayscale
    grayimg <- channel(img_resized,"gray")
    # Path to file
    path <- paste(save_in0, imgname, sep = "")
    # Save image
    writeImage(grayimg, path, quality = 70)
    # Print status
    print(paste("Done",i,sep = " "))},
    # Error function
    error = function(e){print(e)})
}


# Generate a train-test dataset

setwd('C:/Users/Rony/Desktop/DA Project/final_dataset/n0')

# n0 csv file
out_file <- "C:/Users/Rony/Desktop/DA Project/csv_files/n0.csv"

# List images in path
images <- list.files()

# Set up Data Frame
df <- data.frame()

# Set image size. In this case 28x28
img_size <- 28*28

# Set label
label <- 0

# Main loop. Loop over each image
for(i in 1:length(images))
{
  # Read image
  img <- readImage(images[i])
  # Get the image as a matrix
  img_matrix <- img@.Data
  # Coerce to a vector
  img_vector <- as.vector(t(img_matrix))
  # Add label
  vec <- c(label, img_vector)
  # Bind rows
  df <- rbind(df,vec)
  # Print status info
  print(paste("Done ", i, sep = ""))
}

# Set names
names(df) <- c("label", paste("pixel", c(1:img_size)))

# Write out dataset
write.csv(df, out_file, row.names = FALSE)


#------------------------- n1 ---------------------------------#
#Read Images
setwd('C:/Users/Rony/Desktop/DA Project/Dataset/n1')
pics <- list.files(path = ".")

for(i in 1:length(pics))
{
  # Try-catch is necessary since some images
  # may not work.
  result <- tryCatch({
    # Image name
    imgname <- pics[i]
    # Read image
    img <- readImage(imgname)
    # Resize image 28x28
    img_resized <- resize(img, w = 28, h = 28)
    # Set to grayscale
    grayimg <- channel(img_resized,"gray")
    # Path to file
    path <- paste(save_in1, imgname, sep = "")
    # Save image
    writeImage(grayimg, path, quality = 70)
    # Print status
    print(paste("Done",i,sep = " "))},
    # Error function
    error = function(e){print(e)})
}


# Generate a train-test dataset

setwd('C:/Users/Rony/Desktop/DA Project/final_dataset/n1')

# n0 csv file
out_file <- "C:/Users/Rony/Desktop/DA Project/csv_files/n1.csv"

# List images in path
images <- list.files()

# Set up Data Frame
df <- data.frame()

# Set image size. In this case 28x28
img_size <- 28*28

# Set label
label <- 1

# Main loop. Loop over each image
for(i in 1:length(images))
{
  # Read image
  img <- readImage(images[i])
  # Get the image as a matrix
  img_matrix <- img@.Data
  # Coerce to a vector
  img_vector <- as.vector(t(img_matrix))
  # Add label
  vec <- c(label, img_vector)
  # Bind rows
  df <- rbind(df,vec)
  # Print status info
  print(paste("Done ", i, sep = ""))
}

# Set names
names(df) <- c("label", paste("pixel", c(1:img_size)))

# Write out dataset
write.csv(df, out_file, row.names = FALSE)




# --------------------------------------- Test and train split and shuffle

# Load datasets
n0 <- read.csv("C:/Users/Rony/Desktop/DA Project/csv_files/n0.csv")
n1 <- read.csv("C:/Users/Rony/Desktop/DA Project/csv_files/n1.csv")

# Bind rows in a single dataset
new <- rbind(n0, n1)
write.csv(new, "C:/Users/Rony/Desktop/DA Project/csv_files/new.csv",row.names = FALSE)


shuffled <- new[sample(nrow(new)),]
write.csv(shuffled, "C:/Users/Rony/Desktop/DA Project/csv_files/shuffled.csv",row.names = FALSE)


# Shuffle new dataset
train_X <- shuffled[1:200,]

# Train-test split
test_y <- shuffled[201:216,]

# Save train-test datasets
write.csv(train_X, "C:/Users/Rony/Desktop/DA Project/csv_files/train.csv",row.names = FALSE)
write.csv(test_y, "C:/Users/Rony/Desktop/DA Project/csv_files/test.csv",row.names = FALSE)




#----------------------------- Train ---------------------------#

# Load MXNet
library(mxnet)

# Train test datasets
train <- read.csv("C:/Users/Rony/Desktop/DA Project/csv_files/train.csv")
test <- read.csv("C:/Users/Rony/Desktop/DA Project/csv_files/test.csv")


# Fix train and test datasets
train <- data.matrix(train)
train_X <- t(train[,-1])
train_y <- train[,1]
train_array <- train_X
dim(train_array) <- c(28, 28, 1, ncol(train_X))
print(dim(train_array))

test__ <- data.matrix(test)
test_X <- t(test[,-1])
test_y <- test[,1]
print(test_y)
test_array <- test_X
dim(test_array) <- c(28, 28, 1, ncol(test_X))



# Model
data <- mx.symbol.Variable('data')

# 1st convolutional layer 5x5 kernel and 20 filters.
conv_1 <- mx.symbol.Convolution(data= data, kernel = c(5,5), num_filter = 20)
tanh_1 <- mx.symbol.Activation(data= conv_1, act_type = "tanh")
pool_1 <- mx.symbol.Pooling(data = tanh_1, pool_type = "max", kernel = c(2,2), stride = c(2,2))

# 2nd convolutional layer 5x5 kernel and 50 filters.
conv_2 <- mx.symbol.Convolution(data = pool_1, kernel = c(5,5), num_filter = 50)
tanh_2 <- mx.symbol.Activation(data = conv_2, act_type = "tanh")
pool_2 <- mx.symbol.Pooling(data = tanh_2, pool_type = "max", kernel = c(2,2), stride = c(2,2))

# 1st fully connected layer
flat <- mx.symbol.Flatten(data = pool_2)
fcl_1 <- mx.symbol.FullyConnected(data = flat, num_hidden = 500)
tanh_3 <- mx.symbol.Activation(data = fcl_1, act_type = "tanh")

# 2nd fully connected layer
fcl_2 <- mx.symbol.FullyConnected(data = tanh_3, num_hidden = 2)

# Output
NN_model <- mx.symbol.SoftmaxOutput(data = fcl_2)

# Set seed for reproducibility
mx.set.seed(200)

# Device used. Sadly not the GPU :-(
device <- mx.cpu()

# Train on 1200 samples
model <- mx.model.FeedForward.create(NN_model, X = train_array, y = train_y,
                                     ctx = device,
                                     num.round = 30,
                                     array.batch.size = 16,
                                     learning.rate = 0.05,
                                     momentum = 0.9,
                                     wd = 0.00001,
                                     eval.metric = mx.metric.accuracy,
                                     epoch.end.callback = mx.callback.log.train.metric(100))

# Test on 36 samples
predict_probs <- predict(model, test_array)
#print(predict_probs)
predicted_labels <- max.col(t(predict_probs)) - 1
print(predicted_labels)
table(test__[,1], predicted_labels)

sum(diag(table(test__[,1], predicted_labels)))/16