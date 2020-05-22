setwd("/home/j3ssi/MA_Source")
# Plotte Amp Apex
x <- read.delim("batch_size_333_O0.txt", header = TRUE, sep = " ", dec = ".")
y <- read.delim("batch_size_333_O1.txt", header = TRUE, sep = " ", dec = ".")
z<- read.delim("batch_size_333_O2.txt", header = TRUE, sep = " ", dec = ".")
#par(new=TRUE)

# Plotte Amp Apex
plot( x$batchSize,log(x$Time),log="y" ,xlim=c(0, 2040), ylim=c(3,5.5), type="l", col="blue" )
par(new=TRUE)
plot( y$BatchSize,log(y$Time), log="y"  ,xlim=c(0, 2040),ylim=c(3,5.5), type="l", col="black" )
abline(h=3.67)
par(new=TRUE)
plot( z$BatchSize, log(z$Time), log="y" ,xlim=c(0, 2040),ylim=c(3,5.5), type="l", col="deepskyblue" )


#Plotte BatchSize vs Time
a <- read.delim("batch_size_123.txt", header = TRUE, sep = " ", dec = ".")
b <- read.delim("batch_size_233.txt", header = TRUE, sep = " ", dec = ".")
c<- read.delim("batch_size_333.txt", header = TRUE, sep = " ", dec = ".")
d<- read.delim("batch_size_423.txt", header = TRUE, sep = " ", dec = ".")
par(new=FALSE)
plot( a$BatchSize,log(a$Time),xlab="batch size",ylab=log(time),log="y" ,xlim=c(0, 5050), ylim=c(2,8), type="l", col="blue" )
par(new=TRUE)
plot( b$BatchSize,log(b$Time),xlab="",ylab="", log="y"  ,xlim=c(0, 5050),ylim=c(2,8), type="l", col="black" )

par(new=TRUE)
plot( c$batch, log(c$time),xlab="",ylab="", log="y" ,xlim=c(0, 5050),ylim=c(2,8), type="l", col="deepskyblue" )
par(new=TRUE)
plot( d$batch, log(d$time),xlab="",ylab="", log="y" ,xlim=c(0, 5050),ylim=c(2,8), type="l", col="deepskyblue4" )
