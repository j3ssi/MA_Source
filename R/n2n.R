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

baselineS1 <- read.delim("baselineS1.txt", header = TRUE, sep = "\t", dec = ".")
baselineSAcc1 <- baselineS1$ValidAcc.

setwd("/home/j3ssi/MA_Source/output/experimente4/Logs")

deeper1 <- read.delim("deeper.txt", header = TRUE, sep = "\t", dec = ".")



n2nwider1 <- read.delim("wider_lr1.txt", header = TRUE, sep = "\t", dec = ".")


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


setwd("/home/j3ssi/MA_Source/output/experimente4/Logs")


n2nWiderRnd1 <- read.delim("widerRnd1.txt", header = TRUE, sep = "\t", dec = ".")
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
setwd("/home/j3ssi/MA_Source/output/experimente4/Logs/BaselineO")


baselineO1 <- read.delim("baselineO1.txt", header = TRUE, sep = "\t", dec = ".")

boxplot(n2nWiderRnd1$TrainEpochTime.s., baselineO1$TrainEpochTime.s.)
        
n2nWiderAccRnd181 <- n2nWiderRnd1$ValidAcc.
n2nWiderAcc181 <- n2nwider1$ValidAcc., n=195)
plot(n2nWiderAccRnd181, col='blue',xlab='',ylab='',xlim=c(0,180),ylim=c(25,96))

plot(baselineSAcc1, col='blue',xlim=c(0,180),ylim=c(25,96), xlab="Epochen", ylab = "Accuracy")

par(new=TRUE)
plot(n2nwider1$X, col='green',xlim=c(0,180),ylim=c(25,96),xlab='Epochen',ylab='Accuracy')
par(new=TRUE)
plot(baselineO1$ValidAcc.,xlab='',ylab='',xlim=c(0,250),ylim=c(75,96))
par(new=TRUE)


plot(deeper1$ValidAcc., col='green',ylim=c(25,96),xlim=c(0,250),xlab='Epochen',ylab='Accuracy')
abline(h=87.71)




library('caTools')
library('pracma')
x <- seq(from=1, to=180)

f <- gradient(baselineSAcc1)

  
plot(f, type ="l"), xlim =c(20,180),ylim=c(0,1.5))
par(new=TRUE)
abline(h=0.1)
for( l in x){
  
}


plot(x, grad(func=(x,y),x0=x)
x <- 