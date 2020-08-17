#fitte geraden

# s=3
a <- read.delim("Fit3.csv", header = TRUE, sep = ",", dec = ".")
count1 <- c(5306, 9978, 14650, 19322, 23994, 28666, 33338, 38010, 42682, 47354)
batch1 <- c(19776, 13280, 10016, 8021, 6688, 5728, 5024, 4480, 4032, 3648)
coeff1 <- c(0.27, 0.38, 0.49, 0.60, 0.72, 0.83, 0.95, 1.06, 1.18, 1.30)
x <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
plot(a$Blockanzahl,a$Parameteranzahl,xlab ="Anzahl an Blöcken", ylab ="Anzahl an Parametern",main ="Zusammenhang Blockanzahl und Parameteranzahl für ein Stage")
plot(a$Blockanzahl,a$Batchgröße,xlab ="Anzahl an Blöcken", ylab ="Batchgrösse bei konstanter Speicherausnutzung",main ="Zusammenhang Blockanzahl und maximale Batchgrösse für ein Stage")
c<-a$Parameteranzahl/a$Blockanzahl
d <- c/a$Batchgröße

plot(a$Blockanzahl,d,xlab ="Anzahl an Blöcken", ylab ="Quotient aus Parameteranzahl und Blockanzahl",main ="Zusammenhang Blockanzahl und Quotient aus Parameteranzahl und Blockanzahl für ein Stage")
gerade1<-lm(d ~ a$Blockanzahl)
summary(gerade1)
par(new=FALSE)
plot(a$Blockanzahl,d,xlab = "Anzahl an Blöcken", ylab ="Quotient")
par(new=TRUE)
abline( 118.933  ,53.737 , col="red" )

#s=3
library("drc")
count3 <- c(98522, 195738, 292954, 390170, 487386, 584602, 681818, 779034, 876250, 973466)
batch3 <- c()
#batch3 <- c(13824, 8878, 6380, 4987, 4096, 3453, 2976, 2656, 2363, 2156)
coeff3  <- c(7.12, 11.02, 15.31, 19.56, 23.80, 28.22, 32.73, 36.66, 41.20, 45.15)
size <- c(1,2,3,4,5,6,7,8,9,10)
mL <- glm(batch3 ~ count3)

gerade4<-lm(batch3~log(count3))
plot(count3,predict(gerade4),type='l')
points(count3,batch3)
gerade3<-lm(coeff3 ~ size)
summary(gerade3)
gerade3x <-lm (count3 ~ size)
plot(gerade3x)


count3S <- c(98522, 195738, 292954, 390170, 487386)
batch3S <- c(648, 557, 469, 387, 330)
#batch3S <- c(642, 552, 463, 384,256)
coeff3S <- c(152.04, 175.71, 208.21, 252.05, 295.39)
size3S <- c(1,2,3,4,5)
gerade3S<-lm(coeff3S ~ size3S)
