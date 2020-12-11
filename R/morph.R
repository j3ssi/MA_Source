setwd("/home/j3ssi/MA_Source/")

morphFlops <-read.delim("logMorphNet1.txt", header = TRUE, sep = "\t", dec = ".")
par(mar = c(5, 4, 4, 4) + 0.3)  
plot(morphFlops$Regularisierer,type='l',col='green',xlab="Epoche",ylab="Regularisierer")
par(new=TRUE)
plot(morphFlops$Zielgroesse,type='l',pch=17,col='blue',xlab="",ylab="",axes=FALSE)
axis(side = 4, at = pretty(range(morphFlops$Zielgroesse)))      # Add second axis
mtext("Zielgröße", side = 4, line = 3) 


setwd("/home/j3ssi/MA_Source/MorphLogs")


morphFlops1e7 <-read.delim("lambda1e-7_flops.txt", header = TRUE, sep = "\t", dec = ".")
morphFlops2.5e7 <-read.delim("lambda2.5e-7_flops.txt", header = TRUE, sep = "\t", dec = ".")
morphFlops5e7 <-read.delim("lambda5e-7_flops.txt", header = TRUE, sep = "\t", dec = ".")
morphFlops7.5e7 <-read.delim("lambda7.5e-7_flops.txt", header = TRUE, sep = "\t", dec = ".")
morphFlops1e9 <-read.delim("lambda1e-9_flops.txt", header = TRUE, sep = "\t", dec = ".")

lastFlop1e7 <- tail(morphFlops1e7, n=1)
lastFlop2.5e7 <- tail(morphFlops2.5e7, n=1)
lastFlop5e7 <- tail(morphFlops5e7, n=1)
lastFlop7.5e7 <- tail(morphFlops7.5e7, n=1)
lastFlop8.75e8 <- tail(morphFlops8.75e8, n=1)
lastFlop1e9 <- tail(morphFlops1e9, n=1)

xpoints <- c(lastFlop1e7$Regularisierer, 
lastFlop2.5e7$Regularisierer,
lastFlop5e7$Regularisierer,
lastFlop7.5e7$Regularisierer)

ypoints <- c(lastFlop1e7$Top1,
lastFlop2.5e7$Top1,
lastFlop5e7$Top1,
lastFlop7.5e7$Top1)


xpoints <- c(lastFlop1e8$Regularisierer,lastFlop5e8$Regularisierer, 
             lastFlop1e9$Regularisierer)
ypoints <- c( lastFlop1e8$Top1,lastFlop5e8$Top1,
             lastFlop1e9$Top1
             )
plot(xpoints,ypoints,xlab='FLOPs',ylab='Accuracy')
text(ypoints~xpoints,labels=c('1e-7', '2.5e-7', '5e-7', '7.5e-7'), cex= 0.9,pos=1)
'6.25e-8', '7.5e-8', '8.75e-8',
