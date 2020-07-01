library(dplyr)
setwd("/home/j3ssi/MA_Source/output/experimente2/")

prune_lasso005_1 <- read.delim("prune_lasso005_1.txt", header = TRUE, sep = "\t", dec = ".")
prune_lasso005_2 <- read.delim("prune_lasso005_2.txt", header = TRUE, sep = "\t", dec = ".")
prune_lasso005_3 <- read.delim("prune_lasso005_3.txt", header = TRUE, sep = "\t", dec = ".")
prune_lasso005_4 <- read.delim("prune_lasso005_4.txt", header = TRUE, sep = "\t", dec = ".")
prune_lasso005_5 <- read.delim("prune_lasso005_5.txt", header = TRUE, sep = "\t", dec = ".")

prune_lasso005_1Sum <- sum(prune_lasso005_1$TrainEpochTime.s.)
prune_lasso005_2Sum <- sum(prune_lasso005_2$TrainEpochTime.s.)
prune_lasso005_3Sum <- sum(prune_lasso005_3$TrainEpochTime.s.)
prune_lasso005_4Sum <- sum(prune_lasso005_4$TrainEpochTime.s.)
prune_lasso005_5Sum <- sum(prune_lasso005_5$TrainEpochTime.s.)

prune_lasso005 <- c(prune_lasso005_1Sum, prune_lasso005_2Sum, prune_lasso005_3Sum, prune_lasso005_4Sum, prune_lasso005_5Sum)

prune_lasso01_1 <- read.delim("prune_lasso01_1.txt", header = TRUE, sep = "\t", dec = ".")
prune_lasso01_2 <- read.delim("prune_lasso01_2.txt", header = TRUE, sep = "\t", dec = ".")
prune_lasso01_3 <- read.delim("prune_lasso01_3.txt", header = TRUE, sep = "\t", dec = ".")
prune_lasso01_4 <- read.delim("prune_lasso01_4.txt", header = TRUE, sep = "\t", dec = ".")
prune_lasso01_5 <- read.delim("prune_lasso01_5.txt", header = TRUE, sep = "\t", dec = ".")

prune_lasso01_1Sum <- sum(prune_lasso01_1$TrainEpochTime.s.)
prune_lasso01_2Sum <- sum(prune_lasso01_2$TrainEpochTime.s.)
prune_lasso01_3Sum <- sum(prune_lasso01_3$TrainEpochTime.s.)
prune_lasso01_4Sum <- sum(prune_lasso01_4$TrainEpochTime.s.)
prune_lasso01_5Sum <- sum(prune_lasso01_5$TrainEpochTime.s.)

prune_lasso01 <- c(prune_lasso01_1Sum, prune_lasso01_2Sum, prune_lasso01_3Sum, prune_lasso01_4Sum, prune_lasso01_5Sum)

prune_lasso015_1 <- read.delim("prune_lasso015_1.txt", header = TRUE, sep = "\t", dec = ".")
prune_lasso015_2 <- read.delim("prune_lasso015_2.txt", header = TRUE, sep = "\t", dec = ".")
prune_lasso015_3 <- read.delim("prune_lasso015_3.txt", header = TRUE, sep = "\t", dec = ".")
prune_lasso015_4 <- read.delim("prune_lasso015_4.txt", header = TRUE, sep = "\t", dec = ".")
prune_lasso015_5 <- read.delim("prune_lasso015_5.txt", header = TRUE, sep = "\t", dec = ".")

prune_lasso015_1Sum <- sum(prune_lasso015_1$TrainEpochTime.s.)
prune_lasso015_2Sum <- sum(prune_lasso015_2$TrainEpochTime.s.)
prune_lasso015_3Sum <- sum(prune_lasso015_3$TrainEpochTime.s.)
prune_lasso015_4Sum <- sum(prune_lasso015_4$TrainEpochTime.s.)
prune_lasso015_5Sum <- sum(prune_lasso015_5$TrainEpochTime.s.)

prune_lasso015 <- c(prune_lasso015_1Sum, prune_lasso015_2Sum, prune_lasso015_3Sum, prune_lasso015_4Sum, prune_lasso015_5Sum)

prune_lasso02_1 <- read.delim("prune_lasso02_1.txt", header = TRUE, sep = "\t", dec = ".")
prune_lasso02_2 <- read.delim("prune_lasso02_2.txt", header = TRUE, sep = "\t", dec = ".")
prune_lasso02_3 <- read.delim("prune_lasso02_3.txt", header = TRUE, sep = "\t", dec = ".")
prune_lasso02_4 <- read.delim("prune_lasso02_4.txt", header = TRUE, sep = "\t", dec = ".")
prune_lasso02_5 <- read.delim("prune_lasso02_5.txt", header = TRUE, sep = "\t", dec = ".")

prune_lasso02_1Sum <- sum(prune_lasso02_1$TrainEpochTime.s.)
prune_lasso02_2Sum <- sum(prune_lasso02_2$TrainEpochTime.s.)
prune_lasso02_3Sum <- sum(prune_lasso02_3$TrainEpochTime.s.)
prune_lasso02_4Sum <- sum(prune_lasso02_4$TrainEpochTime.s.)
prune_lasso02_5Sum <- sum(prune_lasso02_5$TrainEpochTime.s.)

prune_lasso02 <- c(prune_lasso02_1Sum, prune_lasso02_2Sum, prune_lasso02_3Sum, prune_lasso02_4Sum, prune_lasso02_5Sum)

prune_lasso025_1 <- read.delim("prune_lasso025_1.txt", header = TRUE, sep = "\t", dec = ".")
prune_lasso025_2 <- read.delim("prune_lasso025_2.txt", header = TRUE, sep = "\t", dec = ".")
prune_lasso025_3 <- read.delim("prune_lasso025_3.txt", header = TRUE, sep = "\t", dec = ".")
prune_lasso025_4 <- read.delim("prune_lasso025_4.txt", header = TRUE, sep = "\t", dec = ".")
prune_lasso025_5 <- read.delim("prune_lasso025_5.txt", header = TRUE, sep = "\t", dec = ".")

prune_lasso025_1Sum <- sum(prune_lasso025_1$TrainEpochTime.s.)
prune_lasso025_2Sum <- sum(prune_lasso025_2$TrainEpochTime.s.)
prune_lasso025_3Sum <- sum(prune_lasso025_3$TrainEpochTime.s.)
prune_lasso025_4Sum <- sum(prune_lasso025_4$TrainEpochTime.s.)
prune_lasso025_5Sum <- sum(prune_lasso025_5$TrainEpochTime.s.)

prune_lasso025 <- c(prune_lasso025_1Sum, prune_lasso025_2Sum, prune_lasso025_3Sum, prune_lasso025_4Sum, prune_lasso025_5Sum)

t.test(prune_lasso005, prune_lasso01, alternative = "two.sided", var.equal = FALSE)
t.test(prune_lasso005, prune_lasso015, alternative = "two.sided", var.equal = FALSE)
t.test(prune_lasso005, prune_lasso02, alternative = "two.sided", var.equal = FALSE)
t.test(prune_lasso005, prune_lasso025, alternative = "two.sided", var.equal = FALSE)

t.test(prune_lasso01, prune_lasso015, alternative = "two.sided", var.equal = FALSE)
t.test(prune_lasso01, prune_lasso02, alternative = "two.sided", var.equal = FALSE)
t.test(prune_lasso01, prune_lasso025, alternative = "two.sided", var.equal = FALSE)

t.test(prune_lasso015, prune_lasso02, alternative = "two.sided", var.equal = FALSE)
t.test(prune_lasso015, prune_lasso025, alternative = "two.sided", var.equal = FALSE)

t.test(prune_lasso02, prune_lasso025, alternative = "two.sided", var.equal = FALSE)

boxplot(prune_lasso005_1$TrainEpochTime.s., prune_lasso005_2$TrainEpochTime.s., prune_lasso005_3$TrainEpochTime.s., prune_lasso005_4$TrainEpochTime.s., prune_lasso005_5$TrainEpochTime.s.,
        prune_lasso01_1$TrainEpochTime.s., prune_lasso01_2$TrainEpochTime.s., prune_lasso01_3$TrainEpochTime.s., prune_lasso01_4$TrainEpochTime.s., prune_lasso01_5$TrainEpochTime.s.,
        prune_lasso015_1$TrainEpochTime.s., prune_lasso015_2$TrainEpochTime.s., prune_lasso015_3$TrainEpochTime.s., prune_lasso015_4$TrainEpochTime.s., prune_lasso015_5$TrainEpochTime.s.,
        prune_lasso02_1$TrainEpochTime.s., prune_lasso02_2$TrainEpochTime.s., prune_lasso02_3$TrainEpochTime.s., prune_lasso02_4$TrainEpochTime.s., prune_lasso02_5$TrainEpochTime.s.,
        prune_lasso025_1$TrainEpochTime.s., prune_lasso025_2$TrainEpochTime.s., prune_lasso025_3$TrainEpochTime.s., prune_lasso025_4$TrainEpochTime.s., prune_lasso025_5$TrainEpochTime.s.,
        col=c('powderblue', 'powderblue', 'powderblue', 'powderblue', 'powderblue', 
              'mistyrose', 'mistyrose', 'mistyrose', 'mistyrose', 'mistyrose',
              'lightsalmon', 'lightsalmon', 'lightsalmon', 'lightsalmon', 'lightsalmon',
              'lightpink', 'lightpink', 'lightpink', 'lightpink', 'lightpink',
              'lightsalmon4', 'lightsalmon4', 'lightsalmon4', 'lightsalmon4', 'lightsalmon4'),
        ylim=c(-10,260),
        ylab = "Trainingszeit in Sekunden",
        xlab ="verschiedene Experimente")

legend(0,20, legend = c('Lasso 0.05', 'Lasso 0.1', 'Lasso 0.15', 'Lasso 0.2', 'Lasso 0.25'), 
       col= c('powderblue', 'mistyrose', 'lightsalmon', 'lightpink', 'lightsalmon4'), 
       fill=c('powderblue', 'mistyrose', 'lightsalmon', 'lightpink', 'lightsalmon4'), 
       ncol=3, cex=0.8)

boxplot(prune_lasso005, prune_lasso01, prune_lasso015, prune_lasso02, prune_lasso025,
        ylim=c(30000,70000),
        log = "y",
        ylab = "Summe der Trainingszeit in Sekunden",
        xlab = "verschiedene Experimentengruppen")
axis(at=c(1,2,3,4,5),side =1, labels = c('Lasso 0.05', 'Lasso 0.1', 'Lasso 0.15', 'Lasso 0.2', 'Lasso 0.25'))


lassoAcc005_1 <- prune_lasso005_1$ValidAcc.
lassoAcc005_2 <- prune_lasso005_2$ValidAcc.
lassoAcc005_3 <- prune_lasso005_3$ValidAcc.
lassoAcc005_4 <- prune_lasso005_4$ValidAcc.
lassoAcc005_5 <- prune_lasso005_5$ValidAcc.

lassoAcc005_1 <- tail(lassoAcc005_1, n=1)
lassoAcc005_2 <- tail(lassoAcc005_2, n=1)
lassoAcc005_3 <- tail(lassoAcc005_3, n=1)
lassoAcc005_4 <- tail(lassoAcc005_4, n=1)
lassoAcc005_5 <- tail(lassoAcc005_5, n=1)

lassoAcc005 <- c(lassoAcc005_1, lassoAcc005_2, lassoAcc005_3, lassoAcc005_4, lassoAcc005_5)

lassoAcc01_1 <- prune_lasso01_1$ValidAcc.
lassoAcc01_2 <- prune_lasso01_2$ValidAcc.
lassoAcc01_3 <- prune_lasso01_3$ValidAcc.
lassoAcc01_4 <- prune_lasso01_4$ValidAcc.
lassoAcc01_5 <- prune_lasso01_5$ValidAcc.

lassoAcc01_1 <- tail(lassoAcc01_1, n=1)
lassoAcc01_2 <- tail(lassoAcc01_2, n=1)
lassoAcc01_3 <- tail(lassoAcc01_3, n=1)
lassoAcc01_4 <- tail(lassoAcc01_4, n=1)
lassoAcc01_5 <- tail(lassoAcc01_5, n=1)

lassoAcc01 <- c(lassoAcc01_1, lassoAcc01_2, lassoAcc01_3, lassoAcc01_4, lassoAcc005_5)

lassoAcc015_1 <- prune_lasso015_1$ValidAcc.
lassoAcc015_2 <- prune_lasso015_2$ValidAcc.
lassoAcc015_3 <- prune_lasso015_3$ValidAcc.
lassoAcc015_4 <- prune_lasso015_4$ValidAcc.
lassoAcc015_5 <- prune_lasso015_5$ValidAcc.

lassoAcc015_1 <- tail(lassoAcc015_1, n=1)
lassoAcc015_2 <- tail(lassoAcc015_2, n=1)
lassoAcc015_3 <- tail(lassoAcc015_3, n=1)
lassoAcc015_4 <- tail(lassoAcc015_4, n=1)
lassoAcc015_5 <- tail(lassoAcc015_5, n=1)

lassoAcc015 <- c(lassoAcc015_1, lassoAcc015_2, lassoAcc015_3, lassoAcc015_4, lassoAcc015_5)

lassoAcc02_1 <- prune_lasso02_1$ValidAcc.
lassoAcc02_2 <- prune_lasso02_2$ValidAcc.
lassoAcc02_3 <- prune_lasso02_3$ValidAcc.
lassoAcc02_4 <- prune_lasso02_4$ValidAcc.
lassoAcc02_5 <- prune_lasso02_5$ValidAcc.

lassoAcc02_1 <- tail(lassoAcc02_1, n=1)
lassoAcc02_2 <- tail(lassoAcc02_2, n=1)
lassoAcc02_3 <- tail(lassoAcc02_3, n=1)
lassoAcc02_4 <- tail(lassoAcc02_4, n=1)
lassoAcc02_5 <- tail(lassoAcc02_5, n=1)

lassoAcc02 <- c(lassoAcc02_1, lassoAcc02_2, lassoAcc02_3, lassoAcc02_4, lassoAcc02_5)

lassoAcc025_1 <- prune_lasso015_1$ValidAcc.
lassoAcc025_2 <- prune_lasso015_2$ValidAcc.
lassoAcc025_3 <- prune_lasso015_3$ValidAcc.
lassoAcc025_4 <- prune_lasso015_4$ValidAcc.
lassoAcc025_5 <- prune_lasso015_5$ValidAcc.

lassoAcc025_1 <- tail(lassoAcc025_1, n=1)
lassoAcc025_2 <- tail(lassoAcc025_2, n=1)
lassoAcc025_3 <- tail(lassoAcc025_3, n=1)
lassoAcc025_4 <- tail(lassoAcc025_4, n=1)
lassoAcc025_5 <- tail(lassoAcc025_5, n=1)

lassoAcc025 <- c(lassoAcc025_1,lassoAcc025_2, lassoAcc025_3, lassoAcc025_4, lassoAcc025_5)

t.test(lassoAcc005, lassoAcc01, alternative = "two.sided", var.equal = FALSE)
t.test(lassoAcc005, lassoAcc015, alternative = "two.sided", var.equal = FALSE)
t.test(lassoAcc005, lassoAcc02, alternative = "two.sided", var.equal = FALSE)
t.test(lassoAcc005, lassoAcc025, alternative = "two.sided", var.equal = FALSE)

t.test(lassoAcc01, lassoAcc015, alternative = "two.sided", var.equal = FALSE)
t.test(lassoAcc01, lassoAcc02, alternative = "two.sided", var.equal = FALSE)
t.test(lassoAcc01, lassoAcc025, alternative = "two.sided", var.equal = FALSE)

t.test(lassoAcc015, lassoAcc02, alternative = "two.sided", var.equal = FALSE)
t.test(lassoAcc015, lassoAcc025, alternative = "two.sided", var.equal = FALSE)

t.test(lassoAcc02, lassoAcc025, alternative = "two.sided", var.equal = FALSE)

boxplot(lassoAcc005, lassoAcc01, lassoAcc015, lassoAcc02, lassoAcc025,
        ylim=c(92,93.1),
        ylab = "Accuracy",
        xlab = "verschiedene Experimentengruppen")
axis(at=c(1,2,3,4,5),side =1, labels = c('Lasso 0.05', 'Lasso 0.1', 'Lasso 0.15', 'Lasso 0.2', 'Lasso 0.25'))


prune_reconf2_1 <- read.delim("prune_reconf2_1.txt", header = TRUE, sep = "\t", dec = ".")
prune_reconf2_2 <- read.delim("prune_reconf2_2.txt", header = TRUE, sep = "\t", dec = ".")
prune_reconf2_3 <- read.delim("prune_reconf2_3.txt", header = TRUE, sep = "\t", dec = ".")
prune_reconf2_4 <- read.delim("prune_reconf2_4.txt", header = TRUE, sep = "\t", dec = ".")
prune_reconf2_5 <- read.delim("prune_reconf2_5.txt", header = TRUE, sep = "\t", dec = ".")

prune_reconf2_1Sum <- sum(prune_reconf2_1$TrainEpochTime.s.)
prune_reconf2_2Sum <- sum(prune_reconf2_2$TrainEpochTime.s.)
prune_reconf2_3Sum <- sum(prune_reconf2_3$TrainEpochTime.s.)
prune_reconf2_4Sum <- sum(prune_reconf2_4$TrainEpochTime.s.)
prune_reconf2_5Sum <- sum(prune_reconf2_5$TrainEpochTime.s.)

reconf2_sum <- c(prune_reconf2_1Sum, prune_reconf2_2Sum, prune_reconf2_3Sum, prune_reconf2_4Sum, prune_reconf2_5Sum)

prune_reconf5_1 <- read.delim("prune_reconf5_1.txt", header = TRUE, sep = "\t", dec = ".")
prune_reconf5_2 <- read.delim("prune_reconf5_2.txt", header = TRUE, sep = "\t", dec = ".")
prune_reconf5_3 <- read.delim("prune_reconf5_3.txt", header = TRUE, sep = "\t", dec = ".")
prune_reconf5_4 <- read.delim("prune_reconf5_4.txt", header = TRUE, sep = "\t", dec = ".")
prune_reconf5_5 <- read.delim("prune_reconf5_5.txt", header = TRUE, sep = "\t", dec = ".")

prune_reconf5_1Sum <- sum(prune_reconf5_1$TrainEpochTime.s.)
prune_reconf5_2Sum <- sum(prune_reconf5_2$TrainEpochTime.s.)
prune_reconf5_3Sum <- sum(prune_reconf5_3$TrainEpochTime.s.)
prune_reconf5_4Sum <- sum(prune_reconf5_4$TrainEpochTime.s.)
prune_reconf5_5Sum <- sum(prune_reconf5_5$TrainEpochTime.s.)

reconf5_sum <- c(prune_reconf5_1Sum, prune_reconf5_2Sum, prune_reconf5_3Sum, prune_reconf5_4Sum, prune_reconf5_5Sum)

prune_reconf10_1 <- read.delim("prune_reconf10_1.txt", header = TRUE, sep = "\t", dec = ".")
prune_reconf10_2 <- read.delim("prune_reconf10_2.txt", header = TRUE, sep = "\t", dec = ".")
prune_reconf10_3 <- read.delim("prune_reconf10_3.txt", header = TRUE, sep = "\t", dec = ".")
prune_reconf10_4 <- read.delim("prune_reconf10_4.txt", header = TRUE, sep = "\t", dec = ".")
prune_reconf10_5 <- read.delim("prune_reconf10_5.txt", header = TRUE, sep = "\t", dec = ".")

prune_reconf10_1Sum <- sum(prune_reconf10_1$TrainEpochTime.s.)
prune_reconf10_2Sum <- sum(prune_reconf10_2$TrainEpochTime.s.)
prune_reconf10_3Sum <- sum(prune_reconf10_3$TrainEpochTime.s.)
prune_reconf10_4Sum <- sum(prune_reconf10_4$TrainEpochTime.s.)
prune_reconf10_5Sum <- sum(prune_reconf10_5$TrainEpochTime.s.)

reconf10_sum <- c(prune_reconf10_1Sum, prune_reconf10_2Sum, prune_reconf10_3Sum, prune_reconf10_4Sum, prune_reconf10_5Sum)


boxplot(reconf2_sum, reconf5_sum, reconf10_sum,
        col=c('powderblue', 'mistyrose', 'lightsalmon'),
        ylim=c(29500,34000),
        ylab = "Summe der Trainingszeit in Sekunden",
        xlab ="verschiedene Experimentengruppen")

legend(0,115, legend = c('Rekonfintervall 2 Summe', 'Rekonfintervall 5 Summe', 'Rekonfintervall 10 Summe'), 
       col= c('powderblue', 'mistyrose', 'lightsalmon'), 
       fill=c('powderblue', 'mistyrose', 'lightsalmon'), 
       horiz=TRUE, cex=0.8)

t.test(reconf2, reconf5, alternative = "two.sided", var.equal = FALSE)


boxplot(prune_reconf2_1$TrainEpochTime.s., prune_reconf2_2$TrainEpochTime.s., prune_reconf2_3$TrainEpochTime.s., prune_reconf2_4$TrainEpochTime.s., prune_reconf2_5$TrainEpochTime.s.,
        prune_reconf5_1$TrainEpochTime.s., prune_reconf5_2$TrainEpochTime.s., prune_reconf5_3$TrainEpochTime.s., prune_reconf5_4$TrainEpochTime.s., prune_reconf5_5$TrainEpochTime.s.,
        prune_reconf10_1$TrainEpochTime.s., prune_reconf10_2$TrainEpochTime.s., prune_reconf10_3$TrainEpochTime.s., prune_reconf10_4$TrainEpochTime.s., prune_reconf10_5$TrainEpochTime.s.,
        col=c('powderblue', 'powderblue', 'powderblue', 'powderblue', 'powderblue', 
              'mistyrose', 'mistyrose', 'mistyrose', 'mistyrose', 'mistyrose',
              'lightsalmon', 'lightsalmon', 'lightsalmon', 'lightsalmon', 'lightsalmon'),
        ylim=c(0,250),
        ylab = "Trainingszeit in Sekunden",
        xlab ="verschiedene Experimente")

legend(0,10, legend = c('Rekonfintervall 2', 'Rekonfintervall 5', 'Rekonfintervall 10'), 
       col= c('powderblue', 'mistyrose', 'lightsalmon'), 
       fill=c('powderblue', 'mistyrose', 'lightsalmon'), 
       horiz=TRUE, cex=0.8)


acc21 <- prune_reconf2_1$ValidAcc.
acc22 <- prune_reconf2_2$ValidAcc.
acc23 <- prune_reconf2_3$ValidAcc.
acc24 <- prune_reconf2_4$ValidAcc.
acc25 <- prune_reconf2_5$ValidAcc.

acc21 <- tail(acc21, n=1)
acc22 <- tail(acc22, n=1)
acc23 <- tail(acc23, n=1)
acc24 <- tail(acc24, n=1)
acc25 <- tail(acc25, n=1)

acc51 <- prune_reconf5_1$ValidAcc.
acc52 <- prune_reconf5_2$ValidAcc.
acc53 <- prune_reconf5_3$ValidAcc.
acc54 <- prune_reconf5_4$ValidAcc.
acc55 <- prune_reconf5_5$ValidAcc.

acc51 <- tail(acc51, n=1)
acc52 <- tail(acc52, n=1)
acc53 <- tail(acc53, n=1)
acc54 <- tail(acc54, n=1)
acc55 <- tail(acc55, n=1)

acc101 <- prune_reconf10_1$ValidAcc.
acc102 <- prune_reconf10_2$ValidAcc.
acc103 <- prune_reconf10_3$ValidAcc.
acc104 <- prune_reconf10_4$ValidAcc.
acc105 <- prune_reconf10_5$ValidAcc.

acc101 <- tail(acc101, n=1)
acc102 <- tail(acc102, n=1)
acc103 <- tail(acc103, n=1)
acc104 <- tail(acc104, n=1)
acc105 <- tail(acc105, n=1)

acc2 <- c(acc21, acc22, acc23, acc24, acc25)
acc5 <- c(acc51, acc52, acc53, acc54, acc55)
acc10 <- c(acc101, acc102, acc103, acc104, acc105)

boxplot(acc2, acc5, acc10,
        col=c('powderblue', 'mistyrose', 'lightsalmon'),
        ylim=c(92, 92.6),
        ylab = "Accuracy",
        xlab ="verschiedene Experimentengruppen")

legend(0.4,92.05, legend = c('Accuracy 2', 'Accuracy 5', 'Accuracy 10'), 
       col= c('powderblue', 'mistyrose', 'lightsalmon'), 
       fill=c('powderblue', 'mistyrose', 'lightsalmon'), 
       horiz=TRUE, cex=0.8)


baseline1 <- read.delim("baseline1.txt", header = TRUE, sep = "\t", dec = ".")
baseline2 <- read.delim("baseline2.txt", header = TRUE, sep = "\t", dec = ".")
baseline3 <- read.delim("baseline3.txt", header = TRUE, sep = "\t", dec = ".")
baseline4 <- read.delim("baseline4.txt", header = TRUE, sep = "\t", dec = ".")
baseline5 <- read.delim("baseline5.txt", header = TRUE, sep = "\t", dec = ".")

baseline1Sum <- sum(baseline1$TrainEpochTime.s.)
baseline2Sum <- sum(baseline2$TrainEpochTime.s.)
baseline3Sum <- sum(baseline3$TrainEpochTime.s.)
baseline4Sum <- sum(baseline4$TrainEpochTime.s.)
baseline5Sum <- sum(baseline5$TrainEpochTime.s.)

prunethres11 <- read.delim("prune_thres1_1.txt", header = TRUE, sep = "\t", dec = ".")
prunethres12 <- read.delim("prune_thres1_2.txt", header = TRUE, sep = "\t", dec = ".")
prunethres13 <- read.delim("prune_thres1_3.txt", header = TRUE, sep = "\t", dec = ".")
prunethres14 <- read.delim("prune_thres1_4.txt", header = TRUE, sep = "\t", dec = ".")
prunethres15 <- read.delim("prune_thres1_5.txt", header = TRUE, sep = "\t", dec = ".")

prunethres11Sum <- sum(prunethres11$TrainEpochTime.s.)
prunethres12Sum <- sum(prunethres12$TrainEpochTime.s.)
prunethres13Sum <- sum(prunethres13$TrainEpochTime.s.)
prunethres14Sum <- sum(prunethres14$TrainEpochTime.s.)
prunethres15Sum <- sum(prunethres15$TrainEpochTime.s.)

prunethres21 <- read.delim("prune_thres2_1.txt", header = TRUE, sep = "\t", dec = ".")
prunethres22 <- read.delim("prune_thres2_2.txt", header = TRUE, sep = "\t", dec = ".")
prunethres23 <- read.delim("prune_thres2_3.txt", header = TRUE, sep = "\t", dec = ".")
prunethres24 <- read.delim("prune_thres2_4.txt", header = TRUE, sep = "\t", dec = ".")
prunethres25 <- read.delim("prune_thres2_5.txt", header = TRUE, sep = "\t", dec = ".")

prunethres21Sum <- sum(prunethres21$TrainEpochTime.s.)
prunethres22Sum <- sum(prunethres22$TrainEpochTime.s.)
prunethres23Sum <- sum(prunethres23$TrainEpochTime.s.)
prunethres24Sum <- sum(prunethres24$TrainEpochTime.s.)
prunethres25Sum <- sum(prunethres25$TrainEpochTime.s.)

prunethres31 <- read.delim("prune_thres3_1.txt", header = TRUE, sep = "\t", dec = ".")
prunethres32 <- read.delim("prune_thres3_2.txt", header = TRUE, sep = "\t", dec = ".")
prunethres33 <- read.delim("prune_thres3_3.txt", header = TRUE, sep = "\t", dec = ".")
prunethres34 <- read.delim("prune_thres3_4.txt", header = TRUE, sep = "\t", dec = ".")
prunethres35 <- read.delim("prune_thres3_5.txt", header = TRUE, sep = "\t", dec = ".")

prunethres31Sum <- sum(prunethres31$TrainEpochTime.s.)
prunethres32Sum <- sum(prunethres32$TrainEpochTime.s.)
prunethres33Sum <- sum(prunethres33$TrainEpochTime.s.)
prunethres34Sum <- sum(prunethres34$TrainEpochTime.s.)
prunethres35Sum <- sum(prunethres35$TrainEpochTime.s.)


prunethres41 <- read.delim("prune_thres4_1.txt", header = TRUE, sep = "\t", dec = ".")
prunethres42 <- read.delim("prune_thres4_2.txt", header = TRUE, sep = "\t", dec = ".")
prunethres43 <- read.delim("prune_thres4_3.txt", header = TRUE, sep = "\t", dec = ".")
prunethres44 <- read.delim("prune_thres4_4.txt", header = TRUE, sep = "\t", dec = ".")
prunethres45 <- read.delim("prune_thres4_5.txt", header = TRUE, sep = "\t", dec = ".")

prunethres41Sum <- sum(prunethres41$TrainEpochTime.s.)
prunethres42Sum <- sum(prunethres42$TrainEpochTime.s.)
prunethres43Sum <- sum(prunethres43$TrainEpochTime.s.)
prunethres44Sum <- sum(prunethres44$TrainEpochTime.s.)
prunethres45Sum <- sum(prunethres45$TrainEpochTime.s.)


boxplot(prunethres11$TrainEpochTime.s., prunethres12$TrainEpochTime.s., prunethres13$TrainEpochTime.s., prunethres14$TrainEpochTime.s., prunethres15$TrainEpochTime.s.,
        prunethres21$TrainEpochTime.s., prunethres22$TrainEpochTime.s., prunethres23$TrainEpochTime.s., prunethres24$TrainEpochTime.s.,
        prunethres31$TrainEpochTime.s., prunethres32$TrainEpochTime.s., prunethres33$TrainEpochTime.s., prunethres34$TrainEpochTime.s., prunethres35$TrainEpochTime.s.,
        prunethres41$TrainEpochTime.s., prunethres42$TrainEpochTime.s., prunethres43$TrainEpochTime.s., prunethres44$TrainEpochTime.s., prunethres45$TrainEpochTime.s.,
        
        col=c('powderblue', 'powderblue', 'powderblue', 'powderblue', 'powderblue', 
              'mistyrose', 'mistyrose', 'mistyrose', 'mistyrose',
              'lightsalmon', 'lightsalmon', 'lightsalmon', 'lightsalmon', 'lightsalmon',
              'lightpink', 'lightpink', 'lightpink', 'lightpink', 'lightpink')
        ,ylim=c(0,240),
        ylab = "Trainingszeit in Sekunden",
        xlab="verschiedene Experimente"
        )

legend(0,10, legend = c('thres 0.1 ', 'thres 0.01', 'thres 0.001', 'thres 0.0001'), col= c('powderblue',
                        'mistyrose', 'lightsalmon', 'lightpink'), fill=c('powderblue',
                        'mistyrose', 'lightsalmon', 'lightpink'), horiz=TRUE, cex=0.65)

prunethres1Sum <- c(prunethres11Sum, prunethres12Sum, prunethres13Sum, prunethres14Sum, prunethres15Sum)
prunethres2Sum <- c(prunethres21Sum, prunethres22Sum, prunethres23Sum, prunethres24Sum)
prunethres3Sum <- c(prunethres31Sum, prunethres32Sum, prunethres33Sum, prunethres34Sum, prunethres35Sum)
prunethres4Sum <- c(prunethres41Sum, prunethres42Sum, prunethres43Sum, prunethres44Sum, prunethres45Sum)
par(mar=c(10,17,4,1)+.1)
par(mgp=c(5,1,0))
boxplot(prunethres1Sum, prunethres2Sum, prunethres3Sum, prunethres4Sum, ylab = "Summe der Trainingszeit in Sekunden",las=2,
        names = c('thres 0.1','thres 0.01', 'thres 0.001', 'thres 0.0001') )

accthres11 <- tail(prunethres11$ValidAcc.,n=1)
accthres12 <- tail(prunethres21$ValidAcc.,n=1)
accthres13 <- tail(prunethres31$ValidAcc.,n=1)
accthres14 <- tail(prunethres41$ValidAcc.,n=1)

accthres1 <- c(accthres11, accthres12, accthres13, accthres14)

accthres21 <- tail(prunethres21$ValidAcc.,n=1)
accthres22 <- tail(prunethres22$ValidAcc.,n=1)
accthres23 <- tail(prunethres23$ValidAcc.,n=1)
accthres24 <- tail(prunethres24$ValidAcc.,n=1)

accthres2 <- c(accthres21, accthres22, accthres23, accthres24)

accthres31 <- tail(prunethres31$ValidAcc.,n=1)
accthres32 <- tail(prunethres32$ValidAcc.,n=1)
accthres33 <- tail(prunethres33$ValidAcc.,n=1)
accthres34 <- tail(prunethres34$ValidAcc.,n=1)

accthres3 <- c(accthres31, accthres32, accthres33, accthres34)


accthres41 <- tail(prunethres41$ValidAcc.,n=1)
accthres42 <- tail(prunethres42$ValidAcc.,n=1)
accthres43 <- tail(prunethres43$ValidAcc.,n=1)
accthres44 <- tail(prunethres44$ValidAcc.,n=1)
accthres45 <- tail(prunethres45$ValidAcc.,n=1)

accthres4 <- c(accthres41, accthres42, accthres43, accthres44)


boxplot(accthres1, accthres2, accthres3, accthres4, 
        col=c('powderblue', 'mistyrose', 'lightsalmon', 'lightpink', 'lightsalmon4', 'orange'),ylim=c(90.8,93.2),
        ylab = "Accuracy in Prozent",
        xlab="verschiedene Experimente",
        names = c('thres 0.1','thres 0.01', 'thres 0.001', 'thres 0.0001')
)

prunelr21 <-read.delim("prune_lr2_1.txt", header = TRUE, sep = "\t", dec = ".")
prunelr22 <-read.delim("prune_lr2_2.txt", header = TRUE, sep = "\t", dec = ".")
prunelr23 <-read.delim("prune_lr2_3.txt", header = TRUE, sep = "\t", dec = ".")
prunelr24 <-read.delim("prune_lr2_4.txt", header = TRUE, sep = "\t", dec = ".")
prunelr25 <-read.delim("prune_lr2_5.txt", header = TRUE, sep = "\t", dec = ".")


prunelr11 <-read.delim("prune_lr1_1.txt", header = TRUE, sep = "\t", dec = ".")
prunelr12 <-read.delim("prune_lr1_2.txt", header = TRUE, sep = "\t", dec = ".")
prunelr13 <-read.delim("prune_lr1_3.txt", header = TRUE, sep = "\t", dec = ".")
prunelr14 <-read.delim("prune_lr1_4.txt", header = TRUE, sep = "\t", dec = ".")
prunelr15 <-read.delim("prune_lr1_5.txt", header = TRUE, sep = "\t", dec = ".")

prunelr11Sum <- sum(prunelr11$TrainEpochTime.s.)
prunelr12Sum <- sum(prunelr12$TrainEpochTime.s.)
prunelr13Sum <- sum(prunelr13$TrainEpochTime.s.)
prunelr14Sum <- sum(prunelr14$TrainEpochTime.s.)
prunelr15Sum <- sum(prunelr15$TrainEpochTime.s.)

prunelr1 <- c(prunelr11Sum, prunelr12Sum, prunelr13Sum, prunelr14Sum, prunelr15Sum)

prunelr051 <-read.delim("prune_lr05_1.txt", header = TRUE, sep = "\t", dec = ".")
prunelr052 <-read.delim("prune_lr05_2.txt", header = TRUE, sep = "\t", dec = ".")
prunelr053 <-read.delim("prune_lr05_3.txt", header = TRUE, sep = "\t", dec = ".")
prunelr054 <-read.delim("prune_lr05_4.txt", header = TRUE, sep = "\t", dec = ".")
prunelr055 <-read.delim("prune_lr05_5.txt", header = TRUE, sep = "\t", dec = ".")

prunelr051Sum <- sum(prunelr051$TrainEpochTime.s.)
prunelr052Sum <- sum(prunelr052$TrainEpochTime.s.)
prunelr053Sum <- sum(prunelr053$TrainEpochTime.s.)
prunelr054Sum <- sum(prunelr054$TrainEpochTime.s.)
prunelr055Sum <- sum(prunelr055$TrainEpochTime.s.)

prunelr05 <- c(prunelr051Sum, prunelr052Sum, prunelr053Sum, prunelr054Sum, prunelr055Sum)

prunelr0251 <-read.delim("prune_lr025_1.txt", header = TRUE, sep = "\t", dec = ".")
prunelr0252 <-read.delim("prune_lr025_2.txt", header = TRUE, sep = "\t", dec = ".")
prunelr0253 <-read.delim("prune_lr025_3.txt", header = TRUE, sep = "\t", dec = ".")
prunelr0254 <-read.delim("prune_lr025_4.txt", header = TRUE, sep = "\t", dec = ".")
prunelr0255 <-read.delim("prune_lr025_5.txt", header = TRUE, sep = "\t", dec = ".")

prunelr0251Sum <- sum(prunelr0251$TrainEpochTime.s.)
prunelr0252Sum <- sum(prunelr0252$TrainEpochTime.s.)
prunelr0253Sum <- sum(prunelr0253$TrainEpochTime.s.)
prunelr0254Sum <- sum(prunelr0254$TrainEpochTime.s.)
prunelr0255Sum <- sum(prunelr0255$TrainEpochTime.s.)

prunelr025 <- c(prunelr0251Sum, prunelr0252Sum, prunelr0253Sum, prunelr0254Sum, prunelr0255Sum)


prunelr01251 <-read.delim("prune_lr0125_1.txt", header = TRUE, sep = "\t", dec = ".")
prunelr01252 <-read.delim("prune_lr0125_2.txt", header = TRUE, sep = "\t", dec = ".")
prunelr01253 <-read.delim("prune_lr0125_3.txt", header = TRUE, sep = "\t", dec = ".")
prunelr01254 <-read.delim("prune_lr0125_4.txt", header = TRUE, sep = "\t", dec = ".")
prunelr01255 <-read.delim("prune_lr0125_5.txt", header = TRUE, sep = "\t", dec = ".")

prunelr01251Sum <- sum(prunelr01251$TrainEpochTime.s.)
prunelr01252Sum <- sum(prunelr01252$TrainEpochTime.s.)
prunelr01253Sum <- sum(prunelr01253$TrainEpochTime.s.)
prunelr01254Sum <- sum(prunelr01254$TrainEpochTime.s.)
prunelr01255Sum <- sum(prunelr01255$TrainEpochTime.s.)

prunelr0125 <- c(prunelr01251Sum, prunelr01252Sum, prunelr01253Sum, prunelr01254Sum, prunelr01255Sum)

boxplot(prunelr11$TrainEpochTime.s., prunelr12$TrainEpochTime.s., prunelr13$TrainEpochTime.s., prunelr14$TrainEpochTime.s., prunelr15$TrainEpochTime.s.,
        prunelr051$TrainEpochTime.s., prunelr052$TrainEpochTime.s., prunelr053$TrainEpochTime.s., prunelr054$TrainEpochTime.s., prunelr055$TrainEpochTime.s.,
        prunelr0251$TrainEpochTime.s., prunelr0252$TrainEpochTime.s., prunelr0253$TrainEpochTime.s., prunelr0254$TrainEpochTime.s., prunelr0255$TrainEpochTime.s.,
        prunelr01251$TrainEpochTime.s., prunelr01252$TrainEpochTime.s., prunelr01253$TrainEpochTime.s., prunelr01254$TrainEpochTime.s., prunelr01255$TrainEpochTime.s.,
        col=c('powderblue', 'powderblue', 'powderblue', 'powderblue', 'powderblue', 
              'mistyrose', 'mistyrose', 'mistyrose', 'mistyrose','mistyrose'               
        ),ylim=c(10,250),
        ylab = "Trainingszeit in Sekunden pro Epoche",
        xlab="verschiedene Experimente"
)
legend("bottomleft", legend = c('baseline ohne Synchronisation', 'baseline mit Synchronisation'), col= c('powderblue','mistyrose'),fill=c('powderblue', 'mistyrose'), lty=1:2, horiz=TRUE, cex=0.8)
                                                                                                                                                                          


boxplot(prunelr1, prunelr05, prunelr025, prunelr0125, names = c('lr 0.1', 'lr 0.5', 'lr 0.25', 'lr 0.125'))


plot(baseline1$ValidAcc., col='red', xlim=c(90, 180), ylim=c(86,95), ylab="",xlab="")
par(new=TRUE)
plot(prune11$ValidAcc., col='blue', xlim=c(90, 180), ylim=c(86,95),ylab="Accuracy",xlab="Epochen")
legend("bottomright", legend = c('baseline', 'prunetrain'), col= c('red','blue'), lty=1:2, horiz=TRUE, cex=0.8)


plot(baseline1$ValidAcc. -prune1$ValidAcc., col='blue', xlim=c(90, 180), ylim=c(-5,5))