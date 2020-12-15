setwd("/home/j3ssi/MA_Source/")

morphFlops <-read.delim("logMorphNet1.txt", header = TRUE, sep = "\t", dec = ".")
par(mar = c(5, 4, 4, 4) + 0.3)  
plot(morphFlops$Regularisierer,type='l',col='green',xlab="Epoche",ylab="Regularisierer")
par(new=TRUE)
plot(morphFlops$Zielgroesse,type='l',pch=17,col='blue',xlab="",ylab="",axes=FALSE)
axis(side = 4, at = pretty(range(morphFlops$Zielgroesse)))      # Add second axis
mtext("Zielgröße", side = 4, line = 3) 


setwd("/home/j3ssi/MA_Source/MorphLogs")

morphFlops3e8_1 <-read.delim("lambda3e-8_flops.txt", header = TRUE, sep = "\t", dec = ".")
morphFlops3e8_2 <-read.delim("lambda3e-8_flops2.txt", header = TRUE, sep = "\t", dec = ".")
morphFlops3e8_3 <-read.delim("lambda3e-8_flops3.txt", header = TRUE, sep = "\t", dec = ".")
morphFlops3e8_4 <-read.delim("lambda3e-8_flops4.txt", header = TRUE, sep = "\t", dec = ".")

lastFlop3e8_1 <- tail(morphFlops3e8_1, n=1)
lastFlop3e8_2 <- tail(morphFlops3e8_2, n=1)
lastFlop3e8_3 <- tail(morphFlops3e8_3, n=1)
lastFlop3e8_4 <- tail(morphFlops3e8_4, n=1)


xpoints <- c(lastFlop3e8_1$Zielgroesse, 
             lastFlop3e8_2$Zielgroesse,
             lastFlop3e8_3$Zielgroesse,
             lastFlop3e8_4$Zielgroesse)

ypoints <- c(lastFlop3e8_1$Top1,
             lastFlop3e8_2$Top1,
             lastFlop3e8_3$Top1,
             lastFlop3e8_4$Top1)

plot(xpoints,ypoints,xlab='Zielgröße',ylab='Accuracy')


morphFlops1e8 <-read.delim("lambda1e-8_flops.txt", header = TRUE, sep = "\t", dec = ".")
morphFlops2e8 <-read.delim("lambda2e-8_flops.txt", header = TRUE, sep = "\t", dec = ".")
morphFlops3e8 <-read.delim("lambda3e-8_flops.txt", header = TRUE, sep = "\t", dec = ".")
morphFlops4e8 <-read.delim("lambda4e-8_flops.txt", header = TRUE, sep = "\t", dec = ".")


morphFlops1e9 <-read.delim("lambda1e-9_flops.txt", header = TRUE, sep = "\t", dec = ".")

lastFlop1e8 <- tail(morphFlops1e8, n=1)
lastFlop2e8 <- tail(morphFlops2e8, n=1)
lastFlop3e8 <- tail(morphFlops3e8, n=1)
lastFlop4e8 <- tail(morphFlops4e8, n=1)
lastFlop8.75e8 <- tail(morphFlops8.75e8, n=1)
lastFlop1e9 <- tail(morphFlops1e9, n=1)

xpoints <- c(lastFlop1e8$Zielgroesse, 
lastFlop2e8$Zielgroesse,
lastFlop3e8$Zielgroesse,
lastFlop4e8$Zielgroesse)

ypoints <- c(lastFlop1e8$Top1,
lastFlop2e8$Top1,
lastFlop3e8$Top1,
lastFlop4e8$Top1)


xpoints <- c(lastFlop1e8$Regularisierer,lastFlop5e8$Regularisierer, 
             lastFlop1e9$Regularisierer)
ypoints <- c( lastFlop1e8$Top1,lastFlop5e8$Top1,
             lastFlop1e9$Top1
             )
plot(xpoints,ypoints,xlab='FLOPs',ylab='Accuracy')
text(ypoints~xpoints,labels=c('1e-8', '2e-8', '3e-8', '4e-8'), cex= 0.9,pos=1)




setwd("/home/j3ssi/MA_Source/MorphLogs")

morphSize2e7 <-read.delim("lambda2e-6_size.txt", header = TRUE, sep = "\t", dec = ".")
morphSize3e7 <-read.delim("lambda3e-6_size.txt", header = TRUE, sep = "\t", dec = ".")
morphSize4e7 <-read.delim("lambda4e-6_size.txt", header = TRUE, sep = "\t", dec = ".")
morphSize5e7 <-read.delim("lambda5e-6_size.txt", header = TRUE, sep = "\t", dec = ".")

morphSize2e8 <-read.delim("lambda2e-7_size.txt", header = TRUE, sep = "\t", dec = ".")
morphSize3e8 <-read.delim("lambda3e-7_size.txt", header = TRUE, sep = "\t", dec = ".")
morphSize4e8 <-read.delim("lambda4e-7_size.txt", header = TRUE, sep = "\t", dec = ".")
morphSize5e8 <-read.delim("lambda5e-7_size.txt", header = TRUE, sep = "\t", dec = ".")

lastSize2e7 <- tail(morphSize2e7, n=1)
lastSize3e7 <- tail(morphSize3e7, n=1)
lastSize4e7 <- tail(morphSize4e7, n=1)
lastSize5e7 <- tail(morphSize5e7, n=1)


lastSize2e8 <- tail(morphSize2e8, n=1)
lastSize3e8 <- tail(morphSize3e8, n=1)
lastSize4e8 <- tail(morphSize4e8, n=1)
lastSize5e8 <- tail(morphSize5e8, n=1)


xpoints <- c(lastSize2e7$Zielgroesse, lastSize3e7$Zielgroesse, 
             lastSize4e7$Zielgroesse, lastSize5e7$Zielgroesse,
             lastSize2e8$Zielgroesse, lastSize3e8$Zielgroesse, 
             lastSize4e8$Zielgroesse, lastSize5e8$Zielgroesse)
ypoints <- c( lastSize2e7$Top1, lastSize3e7$Top1,
              lastSize4e7$Top1, lastSize5e7$Top1,
              lastSize2e8$Top1, lastSize3e8$Top1,
              lastSize4e8$Top1, lastSize5e8$Top1)
plot(xpoints,ypoints,xlab='Modellgröße',ylab='Accuracy')
text(ypoints~xpoints,labels=c('2e-6', '3e-6', '4e-6', '5e-6','2e-7', '3e-7', '4e-7', '5e-7'), cex= 0.8,pos=2)
