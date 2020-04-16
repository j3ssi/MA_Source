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

count3 <- c(100858, 198074, 295290, 392506, 489722)
batch3 <- c(18656, 12288, 8000, 5952, 4736)
coeff3  <- c(5.40, 8.06, 12.30, 16.49, 20.68)
size <- c(1,2,3,4,5)

gerade3<-lm(coeff ~ size)
summary(gerade3)

