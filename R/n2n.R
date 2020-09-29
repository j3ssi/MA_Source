setwd("/home/j3ssi/MA_Source/output/experimente4/Logs/Baseline")

baseline1 <- read.delim("baseline1.txt", header = TRUE, sep = "\t", dec = ".")
baseline2 <- read.delim("baseline2.txt", header = TRUE, sep = "\t", dec = ".")
baseline3 <- read.delim("baseline3.txt", header = TRUE, sep = "\t", dec = ".")
baseline4 <- read.delim("baseline4.txt", header = TRUE, sep = "\t", dec = ".")
baseline5 <- read.delim("baseline5.txt", header = TRUE, sep = "\t", dec = ".")

baseline1Sum <- sum(baseline1$TrainEpochTime.s.)/180
baseline2Sum <- sum(baseline2$TrainEpochTime.s.)/180
baseline3Sum <- sum(baseline3$TrainEpochTime.s.)/180
baseline4Sum <- sum(baseline4$TrainEpochTime.s.)/180
baseline5Sum <- sum(baseline5$TrainEpochTime.s.)/180

boxplot(baseline1$TrainEpochTime.s., baseline2$TrainEpochTime.s., baseline3$TrainEpochTime.s., baseline4$TrainEpochTime.s.)

baselineSum1 <- c(baseline1Sum, baseline2Sum, baseline3Sum, baseline4Sum, baseline5Sum)

baselineAcc1 <- c(tail(baseline1$ValidAcc.,n=1), tail(baseline2$ValidAcc.,n=1), tail(baseline3$ValidAcc.,n=1), tail(baseline4$ValidAcc.,n=1), tail(baseline5$ValidAcc.,n=1))


setwd("/home/j3ssi/MA_Source/output/experimente4/Logs/N2NWider")


n2nwider1 <- read.delim("n2nwider11.txt", header = TRUE, sep = "\t", dec = ".")
n2nwider2 <- read.delim("n2nwider12.txt", header = TRUE, sep = "\t", dec = ".")
n2nwider3 <- read.delim("n2nwider13.txt", header = TRUE, sep = "\t", dec = ".")
n2nwider4 <- read.delim("n2nwider14.txt", header = TRUE, sep = "\t", dec = ".")
n2nwider5 <- read.delim("n2nwider15.txt", header = TRUE, sep = "\t", dec = ".")

n2nwider1Sum <- sum(n2nwider1$TrainEpochTime.s.)/180
n2nwider2Sum <- sum(n2nwider2$TrainEpochTime.s.)/180
n2nwider3Sum <- sum(n2nwider3$TrainEpochTime.s.)/180
n2nwider4Sum <- sum(n2nwider4$TrainEpochTime.s.)/180
n2nwider5Sum <- sum(n2nwider5$TrainEpochTime.s.)/180

n2nwiderSum1 <- c(n2nwider1Sum, n2nwider2Sum, n2nwider3Sum, n2nwider4Sum, n2nwider5Sum)

n2nwiderAcc1 <- c(tail(n2nwider1$ValidAcc.,n=1), tail(n2nwider2$ValidAcc.,n=1), tail(n2nwider3$ValidAcc.,n=1), tail(n2nwider4$ValidAcc.,n=1), tail(n2nwider5$ValidAcc.,n=1))


setwd("/home/j3ssi/MA_Source/output/experimente4/Logs/N2NWiderRnd")


n2nWiderRnd1 <- read.delim("n2nWiderRnd11.txt", header = TRUE, sep = "\t", dec = ".")
# n2nWiderRnd2 <- read.delim("n2nWiderRnd12.txt", header = TRUE, sep = "\t", dec = ".")
n2nWiderRnd3 <- read.delim("n2nWiderRnd13.txt", header = TRUE, sep = "\t", dec = ".")
n2nWiderRnd4 <- read.delim("n2nWiderRnd14.txt", header = TRUE, sep = "\t", dec = ".")
n2nWiderRnd5 <- read.delim("n2nWiderRnd15.txt", header = TRUE, sep = "\t", dec = ".")

n2nWiderRnd1Sum <- sum(n2nWiderRnd1$TrainEpochTime.s.)/180
# n2nWiderRnd2Sum <- sum(n2nWiderRnd2$TrainEpochTime.s.)/180
n2nWiderRnd3Sum <- sum(n2nWiderRnd3$TrainEpochTime.s.)/180
n2nWiderRnd4Sum <- sum(n2nWiderRnd4$TrainEpochTime.s.)/180
n2nWiderRnd5Sum <- sum(n2nWiderRnd5$TrainEpochTime.s.)/180

n2nWiderRndSum1 <- c(n2nWiderRnd1Sum,  n2nWiderRnd3Sum, n2nWiderRnd4Sum, n2nWiderRnd5Sum)
# n2nWiderRnd2Sum,
n2nWiderRndAcc1 <- c(tail(n2nWiderRnd1$ValidAcc.,n=1), tail(n2nWiderRnd3$ValidAcc.,n=1), tail(n2nWiderRnd4$ValidAcc.,n=1), tail(n2nWiderRnd5$ValidAcc.,n=1))

# , tail(n2nwider2$ValidAcc.,n=1)

boxplot(baselineAcc1,n2nwiderAcc1, n2nWiderRndAcc1)


n2nWiderAccRnd181 <- tail(n2nWiderRnd1$ValidAcc., n=195)
n2nWiderAcc181 <- tail(n2nwider1$ValidAcc., n=195)
plot(n2nWiderAccRnd181, col='blue',xlab='',ylab='',ylim=c(90,94),xlim=c(90,180))
par(new=TRUE)
plot(n2nWiderAcc181, col='green',xlab='Epochen',ylab='Accuracy',ylim=c(90,94),xlim=c(90,180))
par(new=TRUE)
plot(baseline1$ValidAcc.,col='black',xlim=c(90,180),ylim=c(90,94.0),xlab='',ylab='')
