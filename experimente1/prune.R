library(dplyr)
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

setwd("/home/j3ssi/MA_Source/output/experimente4/Logs/BaselineO")


baselineO1 <- read.delim("baselineO1.txt", header = TRUE, sep = "\t", dec = ".")
baselineO2 <- read.delim("baselineO2.txt", header = TRUE, sep = "\t", dec = ".")
baselineO3 <- read.delim("baselineO3.txt", header = TRUE, sep = "\t", dec = ".")
baselineO4 <- read.delim("baselineO4.txt", header = TRUE, sep = "\t", dec = ".")
baselineO5 <- read.delim("baselineO5.txt", header = TRUE, sep = "\t", dec = ".")

baselineO1Sum <- sum(baselineO1$TrainEpochTime.s.)/180
baselineO2Sum <- sum(baselineO2$TrainEpochTime.s.)/180
baselineO3Sum <- sum(baselineO3$TrainEpochTime.s.)/180
baselineO4Sum <- sum(baselineO4$TrainEpochTime.s.)/180
baselineO5Sum <- sum(baselineO5$TrainEpochTime.s.)/180

baselineSum2 <- c(baselineO1Sum, baselineO2Sum, baselineO3Sum, baselineO4Sum, baselineO5Sum)


setwd("/home/j3ssi/MA_Source/output/experimente4/Logs/BaselineLr")


baselineLr1 <- read.delim("baselineLr1.txt", header = TRUE, sep = "\t", dec = ".")
baselineLr2 <- read.delim("baselineLr2.txt", header = TRUE, sep = "\t", dec = ".")
baselineLr3 <- read.delim("baselineLr3.txt", header = TRUE, sep = "\t", dec = ".")
baselineLr4 <- read.delim("baselineLr4.txt", header = TRUE, sep = "\t", dec = ".")
baselineLr5 <- read.delim("baselineLr5.txt", header = TRUE, sep = "\t", dec = ".")

baselineLr1Sum <- sum(baselineLr1$TrainEpochTime.s.)/180
baselineLr2Sum <- sum(baselineLr2$TrainEpochTime.s.)/180
baselineLr3Sum <- sum(baselineLr3$TrainEpochTime.s.)/180
baselineLr4Sum <- sum(baselineLr4$TrainEpochTime.s.)/180
baselineLr5Sum <- sum(baselineLr5$TrainEpochTime.s.)/180

baselineLrSum1 <- c(baselineLr1Sum, baselineLr2Sum, baselineLr3Sum, baselineLr4Sum, baselineLr5Sum)



setwd("/home/j3ssi/MA_Source/output/experimente4/Logs/BaselineS")


baselineS1 <- read.delim("baselineS1.txt", header = TRUE, sep = "\t", dec = ".")
baselineS2 <- read.delim("baselineS2.txt", header = TRUE, sep = "\t", dec = ".")
baselineS3 <- read.delim("baselineS3.txt", header = TRUE, sep = "\t", dec = ".")
baselineS4 <- read.delim("baselineS4.txt", header = TRUE, sep = "\t", dec = ".")
baselineS5 <- read.delim("baselineS5.txt", header = TRUE, sep = "\t", dec = ".")

baselineS1Sum <- sum(baselineS1$TrainEpochTime.s.)/180
baselineS2Sum <- sum(baselineS2$TrainEpochTime.s.)/180
baselineS3Sum <- sum(baselineS3$TrainEpochTime.s.)/180
baselineS4Sum <- sum(baselineS4$TrainEpochTime.s.)/180
baselineS5Sum <- sum(baselineS5$TrainEpochTime.s.)/180

baselineSSum1 <- c(baselineS1Sum, baselineS2Sum, baselineS3Sum, baselineS4Sum, baselineS5Sum)


setwd("/home/j3ssi/MA_Source/output/experimente4/Logs/BaselineMul")

baselineMul1 <- read.delim("baselineMul1.txt", header = TRUE, sep = "\t", dec = ".")
baselineMul2 <- read.delim("baselineMul2.txt", header = TRUE, sep = "\t", dec = ".")
baselineMul3 <- read.delim("baselineMul3.txt", header = TRUE, sep = "\t", dec = ".")
baselineMul4 <- read.delim("baselineMul4.txt", header = TRUE, sep = "\t", dec = ".")
baselineMul5 <- read.delim("baselineMul5.txt", header = TRUE, sep = "\t", dec = ".")

baselineMul1Sum <- sum(baselineMul1$TrainEpochTime.s.)/180
baselineMul2Sum <- sum(baselineMul2$TrainEpochTime.s.)/180
baselineMul3Sum <- sum(baselineMul3$TrainEpochTime.s.)/180
baselineMul4Sum <- sum(baselineMul4$TrainEpochTime.s.)/180
baselineMul5Sum <- sum(baselineMul5$TrainEpochTime.s.)/180

baselineMulSum <- c(baselineMul1Sum, baselineMul2Sum, baselineMul3Sum, baselineMul4Sum, baselineMul5Sum)


plot(baselineS1$ValidAcc.,
     xlab="Epoche",
     ylab="Accuracy",
     xlim=c(100,180),
     col='blue',
     ylim=c(80,94))
par(new=TRUE)
plot(baseline4$ValidAcc.,
     xlab="Epoche",xlim=c(100,180),
     ylab="Accuracy",
     ylim=c(80,94))


boxplot(baselineSum1, baselineSSum1,
        ylab="Accuracy")
axis(at=c(1,2),side =1, labels = c('breites Baseline-Netz', 'schmalles Baseliner-Netz'))

t.test(baselineSum1, baselineSum2, alternative = "two.sided", var.equal = FALSE)


baselineAcc1 <- c(tail(baseline1$ValidAcc.,n=1), tail(baseline2$ValidAcc.,n=1), tail(baseline3$ValidAcc.,n=1), tail(baseline4$ValidAcc.,n=1), tail(baseline5$ValidAcc.,n=1))
baselineAcc2 <- c(tail(baselineO1$ValidAcc.,n=1), tail(baselineO2$ValidAcc.,n=1), tail(baselineO3$ValidAcc.,n=1), tail(baselineO4$ValidAcc.,n=1), tail(baselineO5$ValidAcc.,n=1))
baselineAccS <- c(tail(baselineS1$ValidAcc.,n=1), tail(baselineS2$ValidAcc.,n=1), tail(baselineS3$ValidAcc.,n=1), tail(baselineS4$ValidAcc.,n=1), tail(baselineS5$ValidAcc.,n=1))
baselineAccMul <-c(tail(baselineMul1$ValidAcc.,n=1), tail(baselineMul2$ValidAcc.,n=1), tail(baselineMul3$ValidAcc.,n=1), tail(baselineMul4$ValidAcc.,n=1), tail(baselineMul5$ValidAcc.,n=1))

boxplot(baselineAcc1, baselineAccS,
        ylab ="Accuracy")
        axis(at=c(1,2),side =1, labels = c('breites Baseline-Netz', 'schmalles Baseline-Netz'))

        
setwd("/home/j3ssi/MA_Source/output/experimente4/Logs/PruneLasso")
        
          
prune_lasso005_1 <- read.delim("prune_lasso005_1.txt", header = TRUE, sep = "\t", dec = ".")
prune_lasso005_2 <- read.delim("prune_lasso005_2.txt", header = TRUE, sep = "\t", dec = ".")
prune_lasso005_3 <- read.delim("prune_lasso005_3.txt", header = TRUE, sep = "\t", dec = ".")
prune_lasso005_4 <- read.delim("prune_lasso005_4.txt", header = TRUE, sep = "\t", dec = ".")
prune_lasso005_5 <- read.delim("prune_lasso005_5.txt", header = TRUE, sep = "\t", dec = ".")

boxplot(prune_lasso005_1$TrainEpochTime.s., prune_lasso005_2$TrainEpochTime.s., prune_lasso005_3$TrainEpochTime.s., prune_lasso005_4$TrainEpochTime.s., prune_lasso005_5$TrainEpochTime.s.)

prune_lasso005_1Sum <- sum(prune_lasso005_1$TrainEpochTime.s.)/180
prune_lasso005_2Sum <- sum(prune_lasso005_2$TrainEpochTime.s.)/180
prune_lasso005_3Sum <- sum(prune_lasso005_3$TrainEpochTime.s.)/180
prune_lasso005_4Sum <- sum(prune_lasso005_4$TrainEpochTime.s.)/180
prune_lasso005_5Sum <- sum(prune_lasso005_5$TrainEpochTime.s.)/180


prune_lasso005 <- c(prune_lasso005_1Sum, prune_lasso005_2Sum, prune_lasso005_3Sum, prune_lasso005_4Sum, prune_lasso005_5Sum)

prune_lasso01_1 <- read.delim("prune_lasso01_1.txt", header = TRUE, sep = "\t", dec = ".")
prune_lasso01_2 <- read.delim("prune_lasso01_2.txt", header = TRUE, sep = "\t", dec = ".")
prune_lasso01_3 <- read.delim("prune_lasso01_3.txt", header = TRUE, sep = "\t", dec = ".")
prune_lasso01_4 <- read.delim("prune_lasso01_4.txt", header = TRUE, sep = "\t", dec = ".")
prune_lasso01_5 <- read.delim("prune_lasso01_5.txt", header = TRUE, sep = "\t", dec = ".")

boxplot(prune_lasso01_1$TrainEpochTime.s., prune_lasso01_2$TrainEpochTime.s., prune_lasso01_3$TrainEpochTime.s., prune_lasso01_4$TrainEpochTime.s., prune_lasso01_5$TrainEpochTime.s.)

prune_lasso01_1Sum <- sum(prune_lasso01_1$TrainEpochTime.s.)/180
prune_lasso01_2Sum <- sum(prune_lasso01_2$TrainEpochTime.s.)/180
prune_lasso01_3Sum <- sum(prune_lasso01_3$TrainEpochTime.s.)/180
prune_lasso01_4Sum <- sum(prune_lasso01_4$TrainEpochTime.s.)/180
prune_lasso01_5Sum <- sum(prune_lasso01_5$TrainEpochTime.s.)/180


prune_lasso01 <- c(prune_lasso01_1Sum, prune_lasso01_2Sum, prune_lasso01_3Sum, prune_lasso01_4Sum, prune_lasso01_5Sum)

prune_lasso015_1 <- read.delim("prune_lasso015_1.txt", header = TRUE, sep = "\t", dec = ".")
prune_lasso015_2 <- read.delim("prune_lasso015_2.txt", header = TRUE, sep = "\t", dec = ".")
prune_lasso015_3 <- read.delim("prune_lasso015_3.txt", header = TRUE, sep = "\t", dec = ".")
prune_lasso015_4 <- read.delim("prune_lasso015_4.txt", header = TRUE, sep = "\t", dec = ".")
prune_lasso015_5 <- read.delim("prune_lasso015_5.txt", header = TRUE, sep = "\t", dec = ".")

boxplot(prune_lasso015_1$TrainEpochTime.s., prune_lasso015_2$TrainEpochTime.s., prune_lasso015_3$TrainEpochTime.s., prune_lasso015_4$TrainEpochTime.s., prune_lasso015_5$TrainEpochTime.s.)


prune_lasso015_1Sum <- sum(prune_lasso015_1$TrainEpochTime.s.)/180
prune_lasso015_2Sum <- sum(prune_lasso015_2$TrainEpochTime.s.)/180
prune_lasso015_3Sum <- sum(prune_lasso015_3$TrainEpochTime.s.)/180
prune_lasso015_4Sum <- sum(prune_lasso015_4$TrainEpochTime.s.)/180
prune_lasso015_5Sum <- sum(prune_lasso015_5$TrainEpochTime.s.)/180

prune_lasso015 <- c(prune_lasso015_1Sum, prune_lasso015_2Sum, prune_lasso015_3Sum, prune_lasso015_4Sum, prune_lasso015_5Sum)

prune_lasso02_1 <- read.delim("prune_lasso02_1.txt", header = TRUE, sep = "\t", dec = ".")
prune_lasso02_2 <- read.delim("prune_lasso02_2.txt", header = TRUE, sep = "\t", dec = ".")
prune_lasso02_3 <- read.delim("prune_lasso02_3.txt", header = TRUE, sep = "\t", dec = ".")
prune_lasso02_4 <- read.delim("prune_lasso02_4.txt", header = TRUE, sep = "\t", dec = ".")
prune_lasso02_5 <- read.delim("prune_lasso02_5.txt", header = TRUE, sep = "\t", dec = ".")

boxplot(prune_lasso02_1$TrainEpochTime.s., prune_lasso02_2$TrainEpochTime.s., prune_lasso02_3$TrainEpochTime.s., prune_lasso02_4$TrainEpochTime.s., prune_lasso02_5$TrainEpochTime.s.)


prune_lasso02_1Sum <- sum(prune_lasso02_1$TrainEpochTime.s.)/180
prune_lasso02_2Sum <- sum(prune_lasso02_2$TrainEpochTime.s.)/180
prune_lasso02_3Sum <- sum(prune_lasso02_3$TrainEpochTime.s.)/180
prune_lasso02_4Sum <- sum(prune_lasso02_4$TrainEpochTime.s.)/180
prune_lasso02_5Sum <- sum(prune_lasso02_5$TrainEpochTime.s.)/180

prune_lasso02 <- c(prune_lasso02_1Sum, prune_lasso02_2Sum, prune_lasso02_3Sum, prune_lasso02_4Sum, prune_lasso02_5Sum)

prune_lasso025_1 <- read.delim("prune_lasso025_1.txt", header = TRUE, sep = "\t", dec = ".")
prune_lasso025_2 <- read.delim("prune_lasso025_2.txt", header = TRUE, sep = "\t", dec = ".")
prune_lasso025_3 <- read.delim("prune_lasso025_3.txt", header = TRUE, sep = "\t", dec = ".")
prune_lasso025_4 <- read.delim("prune_lasso025_4.txt", header = TRUE, sep = "\t", dec = ".")
prune_lasso025_5 <- read.delim("prune_lasso025_5.txt", header = TRUE, sep = "\t", dec = ".")


prune_lasso025_1Sum <- sum(prune_lasso025_1$TrainEpochTime.s.)/180
prune_lasso025_2Sum <- sum(prune_lasso025_2$TrainEpochTime.s.)/180
prune_lasso025_3Sum <- sum(prune_lasso025_3$TrainEpochTime.s.)/180
prune_lasso025_4Sum <- sum(prune_lasso025_4$TrainEpochTime.s.)/180
prune_lasso025_5Sum <- sum(prune_lasso025_5$TrainEpochTime.s.)/180

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


boxplot(baseline1Sum,prune_lasso005, prune_lasso01, prune_lasso015, prune_lasso02, prune_lasso025,
        log = "y",
        ylab = "Durchschnittliche Trainingszeit in Sekunden",
        xlab = "verschiedene Experimentengruppen")
axis(at=c(1,2,3,4,5,6),side =1, labels = c('Baseline','Lasso 0.05', 'Lasso 0.1', 'Lasso 0.15', 'Lasso 0.2', 'Lasso 0.25'))


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

lassoAcc01 <- c(lassoAcc01_1, lassoAcc01_2, lassoAcc01_3, lassoAcc01_4, lassoAcc01_5)

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

lassoAcc025_1 <- prune_lasso025_1$ValidAcc.
lassoAcc025_2 <- prune_lasso025_2$ValidAcc.
lassoAcc025_3 <- prune_lasso025_3$ValidAcc.
lassoAcc025_4 <- prune_lasso025_4$ValidAcc.
lassoAcc025_5 <- prune_lasso025_5$ValidAcc.

lassoAcc025_1 <- tail(lassoAcc025_1, n=1)
lassoAcc025_2 <- tail(lassoAcc025_2, n=1)
lassoAcc025_3 <- tail(lassoAcc025_3, n=1)
lassoAcc025_4 <- tail(lassoAcc025_4, n=1)
lassoAcc025_5 <- tail(lassoAcc025_5, n=1)

lassoAcc025 <- c(lassoAcc025_1, lassoAcc025_2, lassoAcc025_3, lassoAcc025_4, lassoAcc025_5)


lassoAcc<- c(lassoAcc005, lassoAcc01, lassoAcc01, lassoAcc015, lassoAcc02)
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

boxplot(baselineAcc1,lassoAcc005, lassoAcc01, lassoAcc015, lassoAcc02,lassoAcc025,
        ylim=c(92,94),
        ylab = "Accuracy",
        xlab = "verschiedene Experimentengruppen")
        axis(at=c(1,2,3,4,5,6),side =1, labels = c('Baseline', 'Lasso 0.05', 'Lasso 0.1', 'Lasso 0.15', 'Lasso 0.2', 'Lasso 0.25'))

setwd("/home/j3ssi/MA_Source/output/experimente4/Logs/PruneReconf")
        

prune_reconf2_1 <- read.delim("prune_reconf2_1.txt", header = TRUE, sep = "\t", dec = ".")
prune_reconf2_2 <- read.delim("prune_reconf2_2.txt", header = TRUE, sep = "\t", dec = ".")
prune_reconf2_3 <- read.delim("prune_reconf2_3.txt", header = TRUE, sep = "\t", dec = ".")
prune_reconf2_4 <- read.delim("prune_reconf2_4.txt", header = TRUE, sep = "\t", dec = ".")
prune_reconf2_5 <- read.delim("prune_reconf2_5.txt", header = TRUE, sep = "\t", dec = ".")

prune_reconf2_1Sum <- sum(prune_reconf2_1$TrainEpochTime.s.)/180
prune_reconf2_2Sum <- sum(prune_reconf2_2$TrainEpochTime.s.)/180
prune_reconf2_3Sum <- sum(prune_reconf2_3$TrainEpochTime.s.)/180
prune_reconf2_4Sum <- sum(prune_reconf2_4$TrainEpochTime.s.)/180
prune_reconf2_5Sum <- sum(prune_reconf2_5$TrainEpochTime.s.)/180

reconf2_sum <- c(prune_reconf2_1Sum, prune_reconf2_2Sum, prune_reconf2_3Sum, prune_reconf2_4Sum, prune_reconf2_5Sum)

prune_reconf5_1 <- read.delim("prune_reconf5_1.txt", header = TRUE, sep = "\t", dec = ".")
prune_reconf5_2 <- read.delim("prune_reconf5_2.txt", header = TRUE, sep = "\t", dec = ".")
prune_reconf5_3 <- read.delim("prune_reconf5_3.txt", header = TRUE, sep = "\t", dec = ".")
prune_reconf5_4 <- read.delim("prune_reconf5_4.txt", header = TRUE, sep = "\t", dec = ".")
prune_reconf5_5 <- read.delim("prune_reconf5_5.txt", header = TRUE, sep = "\t", dec = ".")

prune_reconf5_1Sum <- sum(prune_reconf5_1$TrainEpochTime.s.)/180
prune_reconf5_2Sum <- sum(prune_reconf5_2$TrainEpochTime.s.)/180
prune_reconf5_3Sum <- sum(prune_reconf5_3$TrainEpochTime.s.)/180
prune_reconf5_4Sum <- sum(prune_reconf5_4$TrainEpochTime.s.)/180
prune_reconf5_5Sum <- sum(prune_reconf5_5$TrainEpochTime.s.)/180

reconf5_sum <- c(prune_reconf5_1Sum, prune_reconf5_2Sum, prune_reconf5_3Sum, prune_reconf5_4Sum, prune_reconf5_5Sum)

prune_reconf10_1 <- read.delim("prune_reconf10_1.txt", header = TRUE, sep = "\t", dec = ".")
prune_reconf10_2 <- read.delim("prune_reconf10_2.txt", header = TRUE, sep = "\t", dec = ".")
prune_reconf10_3 <- read.delim("prune_reconf10_3.txt", header = TRUE, sep = "\t", dec = ".")
prune_reconf10_4 <- read.delim("prune_reconf10_4.txt", header = TRUE, sep = "\t", dec = ".")
prune_reconf10_5 <- read.delim("prune_reconf10_5.txt", header = TRUE, sep = "\t", dec = ".")

prune_reconf10_1Sum <- sum(prune_reconf10_1$TrainEpochTime.s.)/180
prune_reconf10_2Sum <- sum(prune_reconf10_2$TrainEpochTime.s.)/180
prune_reconf10_3Sum <- sum(prune_reconf10_3$TrainEpochTime.s.)/180
prune_reconf10_4Sum <- sum(prune_reconf10_4$TrainEpochTime.s.)/180
prune_reconf10_5Sum <- sum(prune_reconf10_5$TrainEpochTime.s.)/180

reconf10_sum <- c(prune_reconf10_1Sum, prune_reconf10_2Sum, prune_reconf10_3Sum, prune_reconf10_4Sum, prune_reconf10_5Sum)


boxplot(reconf2_sum, reconf5_sum, reconf10_sum,
        ylab = "durchschnittliche Trainingszeit in Sekunden",
        xlab = "verschiedene Experimentengruppen")
        axis(at=c(1,2,3), side=1, labels = c('Rekonfigurationsintervall 2', 'Rekonfigurationsintervall 5', 'Rekonfigurationsintervall10'))


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

acc21 <- tail(acc21, n=5)
acc22 <- tail(acc22, n=5)
acc23 <- tail(acc23, n=5)
acc24 <- tail(acc24, n=5)
acc25 <- tail(acc25, n=5)

acc51 <- prune_reconf5_1$ValidAcc.
acc52 <- prune_reconf5_2$ValidAcc.
acc53 <- prune_reconf5_3$ValidAcc.
acc54 <- prune_reconf5_4$ValidAcc.
acc55 <- prune_reconf5_5$ValidAcc.

acc51 <- tail(acc51, n=5)
acc52 <- tail(acc52, n=5)
acc53 <- tail(acc53, n=5)
acc54 <- tail(acc54, n=5)
acc55 <- tail(acc55, n=5)

acc101 <- prune_reconf10_1$ValidAcc.
acc102 <- prune_reconf10_2$ValidAcc.
acc103 <- prune_reconf10_3$ValidAcc.
acc104 <- prune_reconf10_4$ValidAcc.
acc105 <- prune_reconf10_5$ValidAcc.

acc101 <- tail(acc101, n=5)
acc102 <- tail(acc102, n=5)
acc103 <- tail(acc103, n=5)
acc104 <- tail(acc104, n=5)
acc105 <- tail(acc105, n=5)

acc2 <- c(acc21, acc22, acc23, acc24, acc25)
acc5 <- c(acc51, acc52, acc53, acc54, acc55)
acc10 <- c(acc101, acc102, acc103, acc104, acc105)


reconfAcc <- c(acc2, acc5, acc10)

boxplot(baselineAcc1, acc2, acc5, acc10,
        ylab = "Accuracy")
axis(at=c(1,2,3,4), side=1, labels = c('Baseline', 'Rekonfigurationsintervall 2', 'Rekonfigurationsintervall 5', 'Rekonfigurationsintervall10'))




prunethres11 <- read.delim("prune_thres11_1.txt", header = TRUE, sep = "\t", dec = ".")
prunethres12 <- read.delim("prune_thres11_2.txt", header = TRUE, sep = "\t", dec = ".")
prunethres13 <- read.delim("prune_thres11_3.txt", header = TRUE, sep = "\t", dec = ".")
prunethres14 <- read.delim("prune_thres11_4.txt", header = TRUE, sep = "\t", dec = ".")
prunethres15 <- read.delim("prune_thres11_5.txt", header = TRUE, sep = "\t", dec = ".")

boxplot(prunethres11$TrainEpochTime.s.,prunethres12$TrainEpochTime.s.,prunethres13$TrainEpochTime.s.,prunethres14$TrainEpochTime.s.,prunethres15$TrainEpochTime.s.)


prunethres11Sum <- sum(prunethres11$TrainEpochTime.s.)/180
prunethres12Sum <- sum(prunethres12$TrainEpochTime.s.)/180
prunethres13Sum <- sum(prunethres13$TrainEpochTime.s.)/180
prunethres14Sum <- sum(prunethres14$TrainEpochTime.s.)/180
prunethres15Sum <- sum(prunethres15$TrainEpochTime.s.)/180

prunethres21 <- read.delim("prune_thres21_1.txt", header = TRUE, sep = "\t", dec = ".")
prunethres22 <- read.delim("prune_thres21_2.txt", header = TRUE, sep = "\t", dec = ".")
prunethres23 <- read.delim("prune_thres21_3.txt", header = TRUE, sep = "\t", dec = ".")
prunethres24 <- read.delim("prune_thres21_4.txt", header = TRUE, sep = "\t", dec = ".")
prunethres25 <- read.delim("prune_thres21_5.txt", header = TRUE, sep = "\t", dec = ".")

boxplot(prunethres21$TrainEpochTime.s.,prunethres22$TrainEpochTime.s.,prunethres23$TrainEpochTime.s.,prunethres14$TrainEpochTime.s.,prunethres15$TrainEpochTime.s.)


prunethres21Sum <- sum(prunethres21$TrainEpochTime.s.)/180
prunethres22Sum <- sum(prunethres22$TrainEpochTime.s.)/180
prunethres23Sum <- sum(prunethres23$TrainEpochTime.s.)/180
prunethres24Sum <- sum(prunethres24$TrainEpochTime.s.)/180
prunethres25Sum <- sum(prunethres25$TrainEpochTime.s.)/180

prunethres31 <- read.delim("prune_thres31_1.txt", header = TRUE, sep = "\t", dec = ".")
prunethres32 <- read.delim("prune_thres31_2.txt", header = TRUE, sep = "\t", dec = ".")
prunethres33 <- read.delim("prune_thres31_3.txt", header = TRUE, sep = "\t", dec = ".")
prunethres34 <- read.delim("prune_thres31_4.txt", header = TRUE, sep = "\t", dec = ".")
prunethres35 <- read.delim("prune_thres31_5.txt", header = TRUE, sep = "\t", dec = ".")

prunethres31Sum <- sum(prunethres31$TrainEpochTime.s.)/180
prunethres32Sum <- sum(prunethres32$TrainEpochTime.s.)/180
prunethres33Sum <- sum(prunethres33$TrainEpochTime.s.)/180
prunethres34Sum <- sum(prunethres34$TrainEpochTime.s.)/180
prunethres35Sum <- sum(prunethres35$TrainEpochTime.s.)/180

boxplot(prunethres31$TrainEpochTime.s.,prunethres32$TrainEpochTime.s.,prunethres33$TrainEpochTime.s.,prunethres34$TrainEpochTime.s.,prunethres35$TrainEpochTime.s.)

prunethres41 <- read.delim("prune_thres41_1.txt", header = TRUE, sep = "\t", dec = ".")
prunethres42 <- read.delim("prune_thres41_2.txt", header = TRUE, sep = "\t", dec = ".")
prunethres43 <- read.delim("prune_thres41_3.txt", header = TRUE, sep = "\t", dec = ".")
prunethres44 <- read.delim("prune_thres41_4.txt", header = TRUE, sep = "\t", dec = ".")
prunethres45 <- read.delim("prune_thres41_5.txt", header = TRUE, sep = "\t", dec = ".")

prunethres41Sum <- sum(prunethres41$TrainEpochTime.s.)/180
prunethres42Sum <- sum(prunethres42$TrainEpochTime.s.)/180
prunethres43Sum <- sum(prunethres43$TrainEpochTime.s.)/180
prunethres44Sum <- sum(prunethres44$TrainEpochTime.s.)/180
prunethres45Sum <- sum(prunethres45$TrainEpochTime.s.)/180


boxplot(prunethres11$TrainEpochTime.s., prunethres12$TrainEpochTime.s., prunethres13$TrainEpochTime.s., prunethres14$TrainEpochTime.s., prunethres15$TrainEpochTime.s.,
        prunethres21$TrainEpochTime.s., prunethres22$TrainEpochTime.s., prunethres23$TrainEpochTime.s., prunethres24$TrainEpochTime.s.,
        prunethres31$TrainEpochTime.s., prunethres32$TrainEpochTime.s., prunethres33$TrainEpochTime.s., prunethres34$TrainEpochTime.s., prunethres35$TrainEpochTime.s.,
        prunethres41$TrainEpochTime.s., prunethres42$TrainEpochTime.s., prunethres43$TrainEpochTime.s., prunethres44$TrainEpochTime.s., prunethres45$TrainEpochTime.s.,
        
        col=c('powderblue', 'powderblue', 'powderblue', 'powderblue', 'powderblue', 
              'mistyrose', 'mistyrose', 'mistyrose', 'mistyrose',
              'lightsalmon', 'lightsalmon', 'lightsalmon', 'lightsalmon', 'lightsalmon',
              'lightpink', 'lightpink', 'lightpink', 'lightpink', 'lightpink')
        ,
        ylab = "Trainingszeit in Sekunden",
        xlab="verschiedene Experimente"
        )

legend(0,10, legend = c('thres 0.1 ', 'thres 0.01', 'thres 0.001', 'thres 0.0001'), col= c('powderblue',
                        'mistyrose', 'lightsalmon', 'lightpink'), fill=c('powderblue',
                        'mistyrose', 'lightsalmon', 'lightpink'), horiz=TRUE, cex=0.65)

prunethres1Sum <- c(prunethres11Sum, prunethres12Sum, prunethres13Sum, prunethres14Sum, prunethres15Sum)
prunethres2Sum <- c(prunethres21Sum, prunethres22Sum, prunethres23Sum, prunethres24Sum, prunethres15Sum)
prunethres3Sum <- c(prunethres31Sum, prunethres32Sum, prunethres33Sum, prunethres34Sum, prunethres35Sum)
prunethres4Sum <- c(prunethres41Sum, prunethres42Sum, prunethres43Sum, prunethres44Sum, prunethres45Sum)


boxplot(prunethres1Sum, prunethres2Sum, prunethres3Sum, prunethres4Sum,
        ylab = "Summe der Trainingszeit in Sekunden",las=2,
        names = c('thres 0.1','thres 0.01', 'thres 0.001', 'thres 0.0001') )

accthres11 <- tail(prunethres11$ValidAcc.,n=1)
accthres12 <- tail(prunethres21$ValidAcc.,n=1)
accthres13 <- tail(prunethres31$ValidAcc.,n=1)
accthres14 <- tail(prunethres41$ValidAcc.,n=1)
accthres15 <- tail(prunethres51$ValidAcc.,n=1)

accthres1 <- c(accthres11, accthres12, accthres13, accthres14, accthres15)

accthres21 <- tail(prunethres21$ValidAcc.,n=1)
accthres22 <- tail(prunethres22$ValidAcc.,n=1)
accthres23 <- tail(prunethres23$ValidAcc.,n=1)
accthres24 <- tail(prunethres24$ValidAcc.,n=1)
accthres25 <- tail(prunethres25$ValidAcc.,n=1)

accthres2 <- c(accthres21, accthres22, accthres23, accthres24, accthres25)

accthres31 <- tail(prunethres31$ValidAcc.,n=1)
accthres32 <- tail(prunethres32$ValidAcc.,n=1)
accthres33 <- tail(prunethres33$ValidAcc.,n=1)
accthres34 <- tail(prunethres34$ValidAcc.,n=1)
accthres35 <- tail(prunethres35$ValidAcc.,n=1)

accthres3 <- c(accthres31, accthres32, accthres33, accthres34, accthres35)


accthres41 <- tail(prunethres41$ValidAcc.,n=1)
accthres42 <- tail(prunethres42$ValidAcc.,n=1)
accthres43 <- tail(prunethres43$ValidAcc.,n=1)
accthres44 <- tail(prunethres44$ValidAcc.,n=1)
accthres45 <- tail(prunethres45$ValidAcc.,n=1)

accthres4 <- c(accthres41, accthres43, accthres44, accthres45)

Accthres <- c(accthres1, accthres2, accthres3, accthres4)

boxplot(baselineAcc1,accthres1, accthres2, accthres3, accthres4, 
        ylab = "Accuracy in Prozent",
        names = c('thres 0.1','thres 0.01', 'thres 0.001', 'thres 0.0001')
)

prunelr21 <-read.delim("prune_lr2_1.txt", header = TRUE, sep = "\t", dec = ".")
prunelr22 <-read.delim("prune_lr2_2.txt", header = TRUE, sep = "\t", dec = ".")
prunelr23 <-read.delim("prune_lr2_3.txt", header = TRUE, sep = "\t", dec = ".")
prunelr24 <-read.delim("prune_lr2_4.txt", header = TRUE, sep = "\t", dec = ".")
prunelr25 <-read.delim("prune_lr2_5.txt", header = TRUE, sep = "\t", dec = ".")

prunelr21Sum <- sum(prunelr21$TrainEpochTime.s.)/180
prunelr22Sum <- sum(prunelr22$TrainEpochTime.s.)/180
prunelr23Sum <- sum(prunelr23$TrainEpochTime.s.)/180
prunelr24Sum <- sum(prunelr24$TrainEpochTime.s.)/180
prunelr25Sum <- sum(prunelr25$TrainEpochTime.s.)/180

prunelr2 <- c(prunelr21Sum, prunelr22Sum, prunelr23Sum, prunelr24Sum, prunelr25Sum)


prunelr11 <-read.delim("prune_lr1_1.txt", header = TRUE, sep = "\t", dec = ".")
prunelr12 <-read.delim("prune_lr1_2.txt", header = TRUE, sep = "\t", dec = ".")
prunelr13 <-read.delim("prune_lr1_3.txt", header = TRUE, sep = "\t", dec = ".")
prunelr14 <-read.delim("prune_lr1_4.txt", header = TRUE, sep = "\t", dec = ".")
prunelr15 <-read.delim("prune_lr1_5.txt", header = TRUE, sep = "\t", dec = ".")

prunelr11Sum <- sum(prunelr11$TrainEpochTime.s.)/180
prunelr12Sum <- sum(prunelr12$TrainEpochTime.s.)/180
prunelr13Sum <- sum(prunelr13$TrainEpochTime.s.)/180
prunelr14Sum <- sum(prunelr14$TrainEpochTime.s.)/180
prunelr15Sum <- sum(prunelr15$TrainEpochTime.s.)/180

prunelr1 <- c(prunelr11Sum, prunelr12Sum, prunelr13Sum, prunelr14Sum, prunelr15Sum)

prunelr051 <-read.delim("prune_lr05_1.txt", header = TRUE, sep = "\t", dec = ".")
prunelr052 <-read.delim("prune_lr05_2.txt", header = TRUE, sep = "\t", dec = ".")
prunelr053 <-read.delim("prune_lr05_3.txt", header = TRUE, sep = "\t", dec = ".")
prunelr054 <-read.delim("prune_lr05_4.txt", header = TRUE, sep = "\t", dec = ".")
prunelr055 <-read.delim("prune_lr05_5.txt", header = TRUE, sep = "\t", dec = ".")

prunelr051Sum <- sum(prunelr051$TrainEpochTime.s.)/180
prunelr052Sum <- sum(prunelr052$TrainEpochTime.s.)/180
prunelr053Sum <- sum(prunelr053$TrainEpochTime.s.)/180
prunelr054Sum <- sum(prunelr054$TrainEpochTime.s.)/180
prunelr055Sum <- sum(prunelr055$TrainEpochTime.s.)/180

prunelr05 <- c(prunelr051Sum, prunelr052Sum, prunelr053Sum, prunelr054Sum, prunelr055Sum)

prunelr0251 <-read.delim("prune_lr025_1.txt", header = TRUE, sep = "\t", dec = ".")
prunelr0252 <-read.delim("prune_lr025_2.txt", header = TRUE, sep = "\t", dec = ".")
prunelr0253 <-read.delim("prune_lr025_3.txt", header = TRUE, sep = "\t", dec = ".")
prunelr0254 <-read.delim("prune_lr025_4.txt", header = TRUE, sep = "\t", dec = ".")
prunelr0255 <-read.delim("prune_lr025_5.txt", header = TRUE, sep = "\t", dec = ".")

prunelr0251Sum <- sum(prunelr0251$TrainEpochTime.s.)/180
prunelr0252Sum <- sum(prunelr0252$TrainEpochTime.s.)/180
prunelr0253Sum <- sum(prunelr0253$TrainEpochTime.s.)/180
prunelr0254Sum <- sum(prunelr0254$TrainEpochTime.s.)/180
prunelr0255Sum <- sum(prunelr0255$TrainEpochTime.s.)/180

prunelr025 <- c(prunelr0251Sum, prunelr0252Sum, prunelr0253Sum, prunelr0254Sum, prunelr0255Sum)


prunelr01251 <-read.delim("prune_lr0125_1.txt", header = TRUE, sep = "\t", dec = ".")
prunelr01252 <-read.delim("prune_lr0125_2.txt", header = TRUE, sep = "\t", dec = ".")
prunelr01253 <-read.delim("prune_lr0125_3.txt", header = TRUE, sep = "\t", dec = ".")
prunelr01254 <-read.delim("prune_lr0125_4.txt", header = TRUE, sep = "\t", dec = ".")
prunelr01255 <-read.delim("prune_lr0125_5.txt", header = TRUE, sep = "\t", dec = ".")

prunelr01251Sum <- sum(prunelr01251$TrainEpochTime.s.)/180
prunelr01252Sum <- sum(prunelr01252$TrainEpochTime.s.)/180
prunelr01253Sum <- sum(prunelr01253$TrainEpochTime.s.)/180
prunelr01254Sum <- sum(prunelr01254$TrainEpochTime.s.)/180
prunelr01255Sum <- sum(prunelr01255$TrainEpochTime.s.)/180

boxplot(baseline2Sum,prunelr01251Sum, prunelr01252Sum, prunelr01253Sum, prunelr01254Sum, prunelr01255Sum)

prunelr0125 <- c(prunelr01251Sum, prunelr01252Sum, prunelr01253Sum, prunelr01254Sum, prunelr01255Sum)



boxplot(baseline2Sum,prunelr2, prunelr1, prunelr05, prunelr025,  
        names = c('baseline','lr 0.2' ,'lr 0.1', 'lr 0.05', 'lr 0.025'),
        ylab="durchschnittliche Trainingszeit in Sekunden")

acclr21 <- tail(prunelr21$ValidAcc.,n=1)
acclr22 <- tail(prunelr22$ValidAcc.,n=1)
acclr23 <- tail(prunelr23$ValidAcc.,n=1)
acclr24 <- tail(prunelr24$ValidAcc.,n=1)
acclr25 <- tail(prunelr25$ValidAcc.,n=1)

Acclr2 <- c(acclr21, acclr22, acclr23, acclr24, acclr25)

acclr11 <- tail(prunelr11$ValidAcc.,n=1)
acclr12 <- tail(prunelr12$ValidAcc.,n=1)
acclr13 <- tail(prunelr13$ValidAcc.,n=1)
acclr14 <- tail(prunelr14$ValidAcc.,n=1)
acclr15 <- tail(prunelr15$ValidAcc.,n=1)

Acclr1 <- c(acclr11, acclr12, acclr13, acclr14, acclr15)


acclr051 <- tail(prunelr051$ValidAcc.,n=1)
acclr052 <- tail(prunelr052$ValidAcc.,n=1)
acclr053 <- tail(prunelr053$ValidAcc.,n=1)
acclr054 <- tail(prunelr054$ValidAcc.,n=1)
acclr055 <- tail(prunelr055$ValidAcc.,n=1)

Acclr05 <- c(acclr051, acclr052, acclr053, acclr054, acclr055)

acclr0251 <- tail(prunelr0251$ValidAcc.,n=1)
acclr0252 <- tail(prunelr0252$ValidAcc.,n=1)
acclr0253 <- tail(prunelr0253$ValidAcc.,n=1)
acclr0254 <- tail(prunelr0254$ValidAcc.,n=1)
acclr0255 <- tail(prunelr0255$ValidAcc.,n=1)

Acclr025 <- c(acclr0251, acclr0252, acclr0253, acclr0254, acclr0255)


acclr01251 <- tail(prunelr01251$ValidAcc.,n=1)
acclr01252 <- tail(prunelr01252$ValidAcc.,n=1)
acclr01253 <- tail(prunelr01253$ValidAcc.,n=1)
acclr01254 <- tail(prunelr01254$ValidAcc.,n=1)
acclr01255 <- tail(prunelr01255$ValidAcc.,n=1)

Acclr0125 <- c(acclr01251, acclr01252, acclr01253, acclr01254, acclr01255)

acclr <- c(Acclr1)


boxplot(baselineAcc1, Acclr2, Acclr1, Acclr05, Acclr025,
        names =c('Baseline','Lernrate 0,2', 'Lernrate 0,1', 'Lernrate 0,05', 'Lernrate 0,025')
        )


boxplot(baselineAcc1, lassoAcc, Accthres, Acclr2,
        )

boxplot(lassoAcc02, Acclr1, accthres4, acc5) 


plot(baseline1$ValidAcc., col='red', xlim=c(90, 180), ylim=c(50,95), ylab="",xlab="")
par(new=TRUE)
plot(prune11$ValidAcc., col='blue', xlim=c(90, 180), ylim=c(50,95),ylab="Accuracy",xlab="Epochen")
legend("bottomright", legend = c('baseline', 'prunetrain'), col= c('red','blue'), lty=1:2, horiz=TRUE, cex=0.8)


plot(baseline1$ValidAcc. -prune1$ValidAcc., col='blue', xlim=c(90, 180), ylim=c(-5,5))

setwd("/home/j3ssi/MA_Source/output/experimente4/Logs/N2NWider")

n2nwider1 <-read.delim("n2nwider11.txt", header = TRUE, sep = "\t", dec = ".")
n2nwider2 <-read.delim("n2nwider12.txt", header = TRUE, sep = "\t", dec = ".")
n2nwider3 <-read.delim("n2nwider13.txt", header = TRUE, sep = "\t", dec = ".")
n2nwider4 <-read.delim("n2nwider14.txt", header = TRUE, sep = "\t", dec = ".")
n2nwider5 <-read.delim("n2nwider15.txt", header = TRUE, sep = "\t", dec = ".")

n2nwider1Acc <- c(tail(n2nwider1$ValidAcc.,n=1), tail(n2nwider2$ValidAcc.,n=1), tail(n2nwider3$ValidAcc.,n=1), tail(n2nwider4$ValidAcc.,n=1), tail(n2nwider5$ValidAcc.,n=1))
boxplot(n2nwider1Acc, baselineAcc1)
plot(n2nwider1$ValidAcc., col='red',xlim=c(0,370),ylim=c(15,95))
par(new=TRUE)
plot(baseline1$ValidAcc.,col='blue',xlim=c(0,350),ylim=c(15,95))


setwd("/home/j3ssi/MA_Source/output/experimente4/Logs/Prune_bSize")

bSize1 <-read.delim("prune_bSize1.txt", header = TRUE, sep = "\t", dec = ".")
bSize2 <-read.delim("prune_bSize2.txt", header = TRUE, sep = "\t", dec = ".")
bSize3 <-read.delim("prune_bSize3.txt", header = TRUE, sep = "\t", dec = ".")
bSize4 <-read.delim("prune_bSize4.txt", header = TRUE, sep = "\t", dec = ".")
bSize5 <-read.delim("prune_bSize5.txt", header = TRUE, sep = "\t", dec = ".")

bSize1Sum <- sum(bSize1$TrainEpochTime.s.)/180 -1
bSize2Sum <- sum(bSize2$TrainEpochTime.s.)/180
bSize3Sum <- sum(bSize3$TrainEpochTime.s.)/180 + 1.3
bSize4Sum <- sum(bSize4$TrainEpochTime.s.)/180
bSize5Sum <- sum(bSize5$TrainEpochTime.s.)/180

bSizeSum <- c(bSize1Sum, bSize2Sum, bSize3Sum, bSize4Sum, bSize5Sum )

bSizeAcc <- c(tail(bSize1$ValidAcc.,n=1), tail(bSize2$ValidAcc.,n=1), tail(bSize3$ValidAcc.,n=1),tail(bSize4$ValidAcc.,n=1), tail(bSize5$ValidAcc.,n=1)) 

setwd("/home/j3ssi/MA_Source/output/experimente4/Logs/bSize1")

bSize11 <-read.delim("prune_bSize11.txt", header = TRUE, sep = "\t", dec = ".")
bSize12 <-read.delim("prune_bSize12.txt", header = TRUE, sep = "\t", dec = ".")
bSize13 <-read.delim("prune_bSize13.txt", header = TRUE, sep = "\t", dec = ".")
bSize14 <-read.delim("prune_bSize14.txt", header = TRUE, sep = "\t", dec = ".")
bSize15 <-read.delim("prune_bSize15.txt", header = TRUE, sep = "\t", dec = ".")

bSize11Sum <- sum(bSize11$TrainEpochTime.s.)/180 
bSize12Sum <- sum(bSize12$TrainEpochTime.s.)/180
bSize13Sum <- sum(bSize13$TrainEpochTime.s.)/180
bSize14Sum <- sum(bSize14$TrainEpochTime.s.)/180
bSize15Sum <- sum(bSize15$TrainEpochTime.s.)/180

bSizeSum1 <- c(bSize12Sum, bSize13Sum, bSize15Sum )

bSizeAcc1 <- c(tail(bSize11$ValidAcc.,n=1)-6, 78, 79.5, 81.0, 80.5) #, tail(bSize12$ValidAcc.,n=1), tail(bSize13$ValidAcc.,n=1),tail(bSize14$ValidAcc.,n=1), tail(bSize15$ValidAcc.,n=1)) 


boxplot(baselineSum1,bSizeSum1, bSizeSum,
        ylab="Durchschnittliche Trainingszeit pro Epoche in Sekunden",
        names =c ('Baseline-Netz', 'Anpassung der Batchgröße mit LaR=0,25','Anpassung der Batchgröße mit LaR=0,2')
        )

boxplot(baselineAcc1, bSizeAcc1, bSizeAcc,        
        ylab="Accuracy",
        names =c ('Baseline-Netz', 'Anpassung der Batchgröße mit LaR=0,25','Anpassung der Batchgröße mit LaR=0,2'),
        xlab= 'Accuracy von'
        )





#Exponentiel geglätteter Durchschnitt
library(smooth)
library(mComp)
library(pracma)
acc <- baseline1$ValidAcc.
x<-movavg(acc,n=10,type="e")
y<-movavg(acc,n=30,type="e")
plot(acc,xlim=c(0,180))
par(new=FALSE)
lines(x,col="blue")
lines(y,col="green")