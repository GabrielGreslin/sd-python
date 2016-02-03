# Load library
library(e1071)

# Input
filepath='data/Ozone/ozone.dat'

# Load data
data = read.table(filepath, header=TRUE, sep = ' ')
summary(data)

# Preprocess data
data$STATION = as.numeric(data$STATION)
data$RMH2O = data$RMH2O^2

# Separate data
p_training = 0.8
p_test = 1-p_training
nb_training = p_training * nrow(data)
nb_test = p_test * nrow(data)

random_samples = sample(nrow(data), nb_training)
datappr = data[random_samples, ]
datestr = data[-random_samples, ]
#datappr
#datestr

# Build model
svm.reg = svm(O3obs~.,data=datappr)
plot(tune.svm(O3obs~.,data=datappr, cost=c(1, 1.5, 2, 2.5, 3, 3.5)))