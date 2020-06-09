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
        ylim=c(0,120),
        ylab = "Trainingszeit in Sekunden",
        xlab ="verschiedene Experimente")

legend(0,115, legend = c('Lasso 0.05', 'Lasso 0.1', 'Lasso 0.15', 'Lasso 0.2', 'Lasso 0.25'), 
       col= c('powderblue', 'mistyrose', 'lightsalmon', 'lightpink', 'lightsalmon4'), 
       fill=c('powderblue', 'mistyrose', 'lightsalmon', 'lightpink', 'lightsalmon4'), 
       horiz=TRUE, cex=0.8)



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

baseline11 <- read.delim("baseline11.txt", header = TRUE, sep = "\t", dec = ".")
baseline11 <- read.delim("baseline11.txt", header = TRUE, sep = "\t", dec = ".")



boxplot(baseline1$TrainEpochTime.s., baseline2$TrainEpochTime.s., baseline3$TrainEpochTime.s., baseline4$TrainEpochTime.s., baseline5$TrainEpochTime.s., baseline11$TrainEpochTime.s.,
        col=c('powderblue', 'powderblue', 'powderblue', 'powderblue', 'powderblue', 
              'mistyrose'              
        ),ylim=c(10,60),
        ylab = "Trainingszeit in Sekunden pro Epoche",
        xlab="verschiedene Experimente"
)
legend("bottomleft", legend = c('baseline ohne Synchronisation', 'baseline mit Synchronisation'), col= c('powderblue','mistyrose'),fill=c('powderblue', 'mistyrose'), lty=1:2, horiz=TRUE, cex=0.8)
                                                                                                                                                                          




setwd("/home/j3ssi/MA_Source/output/experimente2/")

prune1x <- read.delim("prune1.txt", header = TRUE, sep = "\t", dec = ".")
prune2x <- read.delim("prune2.txt", header = TRUE, sep = "\t", dec = ".")
prune3x <- read.delim("prune3.txt", header = TRUE, sep = "\t", dec = ".")
prune4x <- read.delim("prune4.txt", header = TRUE, sep = "\t", dec = ".")
prune5x <- read.delim("prune5.txt", header = TRUE, sep = "\t", dec = ".")

prune1Sum <- sum(prune1$TrainEpochTime.s.)
prune2Sum <- sum(prune2$TrainEpochTime.s.)
prune3Sum <- sum(prune3$TrainEpochTime.s.)
prune4Sum <- sum(prune4$TrainEpochTime.s.)
prune5Sum <- sum(prune5$TrainEpochTime.s.)

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

boxplot(prune1x$TrainEpochTime.s., prune2x$TrainEpochTime.s., prune3x$TrainEpochTime.s., prune4x$TrainEpochTime.s. , prune5x$TrainEpochTime.s.)


plot(baseline1$ValidAcc., col='red', xlim=c(90, 180), ylim=c(86,95), ylab="",xlab="")
par(new=TRUE)
plot(prune11$ValidAcc., col='blue', xlim=c(90, 180), ylim=c(86,95),ylab="Accuracy",xlab="Epochen")
legend("bottomright", legend = c('baseline', 'prunetrain'), col= c('red','blue'), lty=1:2, horiz=TRUE, cex=0.8)


plot(baseline1$ValidAcc. -prune1$ValidAcc., col='blue', xlim=c(90, 180), ylim=c(-5,5))