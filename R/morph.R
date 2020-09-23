setwd("/home/j3ssi/MA_Source/")

morphFlops <-read.delim("logMorphNet1.txt", header = TRUE, sep = "\t", dec = ".")
par(mar = c(5, 4, 4, 4) + 0.3)  
plot(morphFlops$Regularisierer,type='l',col='green',xlab="Epoche",ylab="Regularisierer")
par(new=TRUE)
plot(morphFlops$Zielgroesse,type='l',pch=17,col='blue',xlab="",ylab="",axes=FALSE)
axis(side = 4, at = pretty(range(morphFlops$Zielgroesse)))      # Add second axis
mtext("Zielgröße", side = 4, line = 3) 


setwd("/home/j3ssi/MA_Source/MorphLogs")


morphFlopse7 <-read.delim("logMorphNetFlopsE7.txt", header = TRUE, sep = "\t", dec = ".")
morphFlopse7 <-read.delim("logMorphNetFlops3E7.txt", header = TRUE, sep = "\t", dec = ".")
morphFlopse7 <-read.delim("logMorphNetFlops6E7.txt", header = TRUE, sep = "\t", dec = ".")
morphFlopse7 <-read.delim("logMorphNetFlops9E7.txt", header = TRUE, sep = "\t", dec = ".")

morphFlopse8 <-read.delim("logMorphNetFlopsE8.txt", header = TRUE, sep = "\t", dec = ".")
morphFlopse9 <-read.delim("logMorphNetFlopsE9.txt", header = TRUE, sep = "\t", dec = ".")

lastFlope7 <- tail(morphFlopse7,n=1)
lastFlope8 <- tail(morphFlopse8,n=1)
lastFlope9 <- tail(morphFlopse9,n=1)


xpoints <- c(lastFlope7$Zielgroesse, lastFlope8$Zielgroesse, lastFlope9$Zielgroesse)
ypoints <- c(lastFlope7$Top1, lastFlope8$Top1, lastFlope9$Top1)
plot(xpoints,ypoints)