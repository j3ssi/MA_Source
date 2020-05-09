#fitte geraden

# s=1
count1 <- c(5306, 9978, 14650, 19322, 23994, 28666, 33338, 38010, 42682, 47354)
batch1 <- c(19776, 13280, 10016, 8021, 6688, 5728, 5024, 4480, 4032, 3648)
coeff1 <- c(0.27, 0.38, 0.49, 0.60, 0.72, 0.83, 0.95, 1.06, 1.18, 1.30)
x <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
plot(x,count1,xlab ="Anzahl an Blöcken", ylab ="Anzahl an Parametern",main ="Zusammenhang Blockanzahl und Parameteranzahl für ein Stage")
plot(x,batch1,xlab ="Anzahl an Blöcken", ylab ="maximale Batchgrösse",main ="Zusammenhang Blockanzahl und maximale Batchgrösse für ein Stage")
c<-count1/x

plot(x,c,xlab ="Anzahl an Blöcken", ylab ="Quotient aus Parameteranzahl und Blockanzahl",main ="Zusammenhang Blockanzahl und Quotient aus Parameteranzahl und Blockanzahl für ein Stage")
gerade1<-lm(coeff1 ~ x)
summary(gerade1)
par(new=FALSE)
plot(x,coeff1,xlim=c(1,10),ylim=c(0,1.5))
par(new=TRUE)
abline( 0.1487  ,    0.1144 , col="red" )

#s=3

count3 <- c(98522, 195738, 292954, 390170, 487386)
batch3 <- c(13856, 8896, 6380, 4987, 4096)
coeff3  <- c(7.11, 11.00, 15.31, 19.56, 23.80)
size <- c(1,2,3,4,5)

gerade3<-lm(coeff ~ size)
summary(gerade3)
gerade3x <-lm (count3 ~ size)
plot(gerade3x)
