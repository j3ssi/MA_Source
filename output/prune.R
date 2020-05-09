setwd("/home/j3ssi/MA_Source/output")


prune1 <- read.delim("prune1.txt")
prune1Time<- prune1$TrainEpochTime.s.

prune2 <- read.delim("prune2.txt")
prune2Time<- prune2$TrainEpochTime.s.

prune3 <- read.delim("prune3.txt")
prune3Time<- prune3$TrainEpochTime.s.

prune4 <- read.delim("prune4.txt")
prune4Time<- prune4$TrainEpochTime.s.

prune5 <- read.delim("prune5.txt")
prune5Time<- prune5$TrainEpochTime.s.

prune6 <- read.delim("prune6.txt")
prune6Time<- prune6$TrainEpochTime.s.

prune7 <- read.delim("prune7.txt")
prune7Time<- prune7$TrainEpochTime.s.

prune8 <- read.delim("prune8.txt")
prune8Time<- prune8$TrainEpochTime.s.

prune9 <- read.delim("prune9.txt")
prune9Time<- prune9$TrainEpochTime.s.

prune10 <- read.delim("prune10.txt")
prune10Time<- prune10$TrainEpochTime.s.

prune11 <- read.delim("prune11.txt")
prune11Time<- prune11$TrainEpochTime.s.

prune12 <- read.delim("prune12.txt")
prune12Time<- prune12$TrainEpochTime.s.

prune13 <- read.delim("prune13.txt")
prune13Time<- prune13$TrainEpochTime.s.

prune14 <- read.delim("prune14.txt")
prune14Time<- prune14$TrainEpochTime.s.

prune15 <- read.delim("prune15.txt")
prune15Time<- prune15$TrainEpochTime.s.


baseline1 <- read.delim("baseline1.txt")
baseline1Time<- baseline1$TrainEpochTime.s.

baseline2 <- read.delim("baseline2.txt")
baseline2Time<- baseline2$TrainEpochTime.s.

baseline3 <- read.delim("baseline3.txt")
baseline3Time<- baseline3$TrainEpochTime.s.

baseline4 <- read.delim("baseline4.txt")
baseline4Time<- baseline4$TrainEpochTime.s.

baseline5 <- read.delim("baseline5.txt")
baseline5Time<- baseline5$TrainEpochTime.s.

boxplot(baseline1Time, baseline2Time, baseline3Time, baseline4Time,baseline5Time,
        prune1Time, prune2Time, prune3Time, prune4Time, prune5Time, 
        prune6Time, prune7Time, prune8Time, prune9Time, prune10Time,
        prune11Time, prune12Time, prune13Time, prune14Time, prune15Time)

