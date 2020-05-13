setwd("/home/j3ssi/MA_Source/output")

prune1 <- read.delim("prune1.txt", header = TRUE, sep = "\t", dec = ".")
prune2 <- read.delim("prune2.txt", header = TRUE, sep = "\t", dec = ".")
prune3 <- read.delim("prune3.txt", header = TRUE, sep = "\t", dec = ".")
prune4 <- read.delim("prune4.txt", header = TRUE, sep = "\t", dec = ".")
prune5 <- read.delim("prune5.txt", header = TRUE, sep = "\t", dec = ".")

prune1Sum <- sum(prune1$TrainEpochTime.s.)
prune2Sum <- sum(prune2$TrainEpochTime.s.)
prune3Sum <- sum(prune3$TrainEpochTime.s.)
prune4Sum <- sum(prune4$TrainEpochTime.s.)
prune5Sum <- sum(prune5$TrainEpochTime.s.)


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

prune6 <- read.delim("prune6.txt", header = TRUE, sep = "\t", dec = ".")
prune7 <- read.delim("prune7.txt", header = TRUE, sep = "\t", dec = ".")
prune8 <- read.delim("prune8.txt", header = TRUE, sep = "\t", dec = ".")
prune9 <- read.delim("prune9.txt", header = TRUE, sep = "\t", dec = ".")
prune10 <- read.delim("prune10.txt", header = TRUE, sep = "\t", dec = ".")

prune6Sum <- sum(prune6$TrainEpochTime.s.)
prune7Sum <- sum(prune7$TrainEpochTime.s.)
prune8Sum <- sum(prune8$TrainEpochTime.s.)
prune9Sum <- sum(prune9$TrainEpochTime.s.)
prune10Sum <- sum(prune10$TrainEpochTime.s.)

prune11 <- read.delim("prune11.txt", header = TRUE, sep = "\t", dec = ".")
prune12 <- read.delim("prune12.txt", header = TRUE, sep = "\t", dec = ".")
prune13 <- read.delim("prune13.txt", header = TRUE, sep = "\t", dec = ".")
prune14 <- read.delim("prune14.txt", header = TRUE, sep = "\t", dec = ".")
prune15 <- read.delim("prune15.txt", header = TRUE, sep = "\t", dec = ".")

prune11Sum <- sum(prune11$TrainEpochTime.s.)
prune12Sum <- sum(prune12$TrainEpochTime.s.)
prune13Sum <- sum(prune13$TrainEpochTime.s.)
prune14Sum <- sum(prune14$TrainEpochTime.s.)
prune15Sum <- sum(prune15$TrainEpochTime.s.)


prune31 <- read.delim("prune31.txt", header = TRUE, sep = "\t", dec = ".")
prune32 <- read.delim("prune32.txt", header = TRUE, sep = "\t", dec = ".")
prune33 <- read.delim("prune33.txt", header = TRUE, sep = "\t", dec = ".")
prune34 <- read.delim("prune34.txt", header = TRUE, sep = "\t", dec = ".")
prune35 <- read.delim("prune35.txt", header = TRUE, sep = "\t", dec = ".")

prune36 <- read.delim("prune36.txt", header = TRUE, sep = "\t", dec = ".")
prune37 <- read.delim("prune37.txt", header = TRUE, sep = "\t", dec = ".")
prune38 <- read.delim("prune38.txt", header = TRUE, sep = "\t", dec = ".")
prune39 <- read.delim("prune39.txt", header = TRUE, sep = "\t", dec = ".")
prune40 <- read.delim("prune40.txt", header = TRUE, sep = "\t", dec = ".")



boxplot(baseline1$TrainEpochTime.s., baseline2$TrainEpochTime.s., baseline3$TrainEpochTime.s., baseline4$TrainEpochTime.s., baseline5$TrainEpochTime.s.,
        prune1$TrainEpochTime.s., prune2$TrainEpochTime.s., prune3$TrainEpochTime.s., prune4$TrainEpochTime.s., prune5$TrainEpochTime.s., 
        prune6$TrainEpochTime.s., prune7$TrainEpochTime.s., prune8$TrainEpochTime.s., prune9$TrainEpochTime.s., prune10$TrainEpochTime.s.,
        prune11$TrainEpochTime.s., prune12$TrainEpochTime.s., prune13$TrainEpochTime.s., prune14$TrainEpochTime.s., prune15$TrainEpochTime.s.,
        prune31$TrainEpochTime.s., prune32$TrainEpochTime.s., prune33$TrainEpochTime.s., prune34$TrainEpochTime.s., prune35$TrainEpochTime.s.,
        prune36$TrainEpochTime.s., prune37$TrainEpochTime.s., prune38$TrainEpochTime.s., prune39$TrainEpochTime.s., prune40$TrainEpochTime.s.,
        
        col=c('powderblue', 'powderblue', 'powderblue', 'powderblue', 'powderblue', 
              'mistyrose', 'mistyrose', 'mistyrose', 'mistyrose', 'mistyrose',
              'lightsalmon', 'lightsalmon', 'lightsalmon', 'lightsalmon', 'lightsalmon',
              'lightpink', 'lightpink', 'lightpink', 'lightpink', 'lightpink',
              'lightsalmon4', 'lightsalmon4', 'lightsalmon4', 'lightsalmon4', 'lightsalmon4',       
              'orange', 'orange', 'orange', 'orange', 'orange'              
              ),ylim=c(0,120),
        ylab = "Trainingszeit in Sekunden",
        xlab="verschiedene Experimente"
        )

legend(0,115, legend = c('baseline', 'prune1', 'prune2', 'prune3', 'prune4', 'prune5'), col= c('powderblue',
                        'mistyrose', 'lightsalmon', 'lightpink', 'lightsalmon4', 'orange'), fill=c('powderblue',
                                                                                                   'mistyrose', 'lightsalmon', 'lightpink', 'lightsalmon4', 'orange'), horiz=TRUE, cex=0.8)

prune36Sum <- sum(prune36$TrainEpochTime.s.)
prune37Sum <- sum(prune37$TrainEpochTime.s.)
prune38Sum <- sum(prune38$TrainEpochTime.s.)
prune39Sum <- sum(prune39$TrainEpochTime.s.)
prune40Sum <- sum(prune40$TrainEpochTime.s.)


prune <- c(prune36Sum, prune37Sum, prune38Sum, prune39Sum, prune40Sum)
baseline <- c(baseline1Sum, baseline2Sum, baseline3Sum, baseline4Sum, baseline5Sum)
boxplot(baseline, prune, col=c('blue', 'orange'),ylab = "Summe der Trainingszeit in Sekunden",
xlab="verschiedene Experimente")

legend("bottomleft", legend = c('baseline', 'prune5'), col= c('blue', 'orange'), fill=c('blue', 'orange'), horiz=TRUE, cex=0.8)


baselineAcc <- c(baseline1$ValidAcc.[180], baseline2$ValidAcc.[180], baseline3$ValidAcc.[180], baseline4$ValidAcc.[180], baseline5$ValidAcc.[180])

pruneAcc1 <- c(prune1$ValidAcc.[180], prune2$ValidAcc.[180], prune3$ValidAcc.[180], prune4$ValidAcc.[180],prune5$ValidAcc.[180])

pruneAcc2 <- c(prune6$ValidAcc.[180], prune7$ValidAcc.[180], prune8$ValidAcc.[180], prune9$ValidAcc.[180],prune10$ValidAcc.[180])

pruneAcc3 <- c(prune11$ValidAcc.[180], prune12$ValidAcc.[180], prune13$ValidAcc.[180], prune14$ValidAcc.[180], prune15$ValidAcc.[180])

pruneAcc4 <- c(prune31$ValidAcc.[180], prune32$ValidAcc.[180], prune33$ValidAcc.[180], prune34$ValidAcc.[180], prune35$ValidAcc.[180])

pruneAcc5 <- c(prune36$ValidAcc.[180], prune37$ValidAcc.[180], prune38$ValidAcc.[180], prune39$ValidAcc.[180], prune40$ValidAcc.[180])

boxplot(baselineAcc, pruneAcc1, pruneAcc2, pruneAcc3, pruneAcc4, pruneAcc5, 
        col=c('powderblue', 'mistyrose', 'lightsalmon', 'lightpink', 'lightsalmon4', 'orange'),ylim=c(90.8,93.2),
        ylab = "Accuracy in Prozent",
        xlab="verschiedene Experimente"
)

legend("bottomleft", legend = c('baseline', 'prune1', 'prune2', 'prune3', 'prune4', 'prune5'), col= c('powderblue', 'mistyrose', 'lightsalmon', 'lightpink', 'lightsalmon4', 'orange'), fill=c('powderblue', 'mistyrose', 'lightsalmon', 'lightpink', 'lightsalmon4', 'orange'), horiz=TRUE, cex=0.8)

