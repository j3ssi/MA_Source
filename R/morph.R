setwd("/home/j3ssi/MA_Source/")

morphFlops <-read.delim("logMorphNet1.txt", header = TRUE, sep = "\t", dec = ".")
par(mar = c(5, 4, 4, 4) + 0.3)  
plot(morphFlops$Regularisierer,type='l',col='green',xlab="Epoche",ylab="Regularisierer")
par(new=TRUE)
plot(morphFlops$Zielgroesse,type='l',pch=17,col='blue',xlab="",ylab="",axes=FALSE)
axis(side = 4, at = pretty(range(morphFlops$Zielgroesse)))      # Add second axis
mtext("Zielgröße", side = 4, line = 3) 


setwd("/home/j3ssi/MA_Source/MorphLogs")


morphFlopse8 <-read.delim("logMorphNetFlopsE8.txt", header = TRUE, sep = "\t", dec = ".")
morphFlops3e8 <-read.delim("logMorphNetFlops3E8.txt", header = TRUE, sep = "\t", dec = ".")
morphFlops6e8 <-read.delim("logMorphNetFlops6E8.txt", header = TRUE, sep = "\t", dec = ".")
morphFlops9e8 <-read.delim("logMorphNetFlops9E8.txt", header = TRUE, sep = "\t", dec = ".")
morphFlopse9 <-read.delim("logMorphNetFlopsE9.txt", header = TRUE, sep = "\t", dec = ".")

lastFlope8 <- tail(morphFlopse8, n=1)
lastFlops3e8 <- tail(morphFlops3e8, n=1)
lastFlop6e8 <- tail(morphFlops6e8, n=1)
lastFlop9e8 <- tail(morphFlops9e8, n=1)
lastFlope9 <- tail(morphFlopse9, n=1)


xpoints <- c(lastFlop6e8$Zielgroesse, 
             lastFlop9e8$Zielgroesse, lastFlope9$Zielgroesse)
ypoints <- c( lastFlop6e8$Top1,
             lastFlop9e8$Top1,
             lastFlope9$Top1)
plot(xpoints,ypoints)
text(ypoints~xpoints,labels=c('6e-8', '9e-8', 'e-9'), cex= 0.9,pos=1)

