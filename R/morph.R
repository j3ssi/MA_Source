setwd("/home/j3ssi/MA_Source/")

morphFlops <-read.delim("logMorphNet1.txt", header = TRUE, sep = "\t", dec = ".")
par(mar = c(5, 4, 4, 4) + 0.3)  
plot(morphFlops$Regularisierer,type='l',col='green',xlab="Epoche",ylab="Regularisierer")
par(new=TRUE)
plot(morphFlops$Zielgroesse,type='l',pch=17,col='blue',xlab="",ylab="",axes=FALSE)
axis(side = 4, at = pretty(range(morphFlops$Zielgroesse)))      # Add second axis
mtext("Zielgröße", side = 4, line = 3) 


setwd("/home/j3ssi/MA_Source/MorphLogs")


morphFlops5e8 <-read.delim("lbda5e-8_flops.txt", header = TRUE, sep = "\t", dec = ".")
morphFlops6.25e8 <-read.delim("lbda6.25e-8_flops.txt", header = TRUE, sep = "\t", dec = ".")
morphFlops7.5e8 <-read.delim("lbda7.5e-8_flops.txt", header = TRUE, sep = "\t", dec = ".")
morphFlops8.75e8 <-read.delim("lbda8.75e-8_flops.txt", header = TRUE, sep = "\t", dec = ".")
morphFlops1e9 <-read.delim("lbda1e-9_flops.txt", header = TRUE, sep = "\t", dec = ".")

lastFlop5e8 <- tail(morphFlops5e8, n=1)
lastFlop6.25e8 <- tail(morphFlops6.25e8, n=1)
lastFlop7.5e8 <- tail(morphFlops7.5e8, n=1)
lastFlop8.75e8 <- tail(morphFlops8.75e8, n=1)
lastFlop1e9 <- tail(morphFlops1e9, n=1)


xpoints <- c(lastFlop5e8$Regularisierer, 
             lastFlop6.25e8$Regularisierer, 
             lastFlop7.5e8$Regularisierer,
             lastFlop8.75e8$Regularisierer,
             lastFlop1e9$Regularisierer)
ypoints <- c( lastFlop5e8$Top1,
             lastFlop6.25e8$Top1,
             lastFlop7.5e8$Top1,
             lastFlop8.75e8$Top1,
             lastFlop1e9$Top1
             )
plot(xpoints,ypoints,xlab='FLOPs',ylab='Accuracy')
text(ypoints~xpoints,labels=c('5e-8', '6.25e-8', '7.5e-8', '8.75e-8', '1e-9'), cex= 0.9,pos=1)

