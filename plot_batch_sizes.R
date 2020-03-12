x <- read.delim("batch_size_333.txt", header = TRUE, sep = " ", dec = ".")
y <- read.delim("batch_size_333_fp16.txt", header = TRUE, sep = " ", dec = ".")

plot( x$batch, log(x$time) ,xlim=c(0, 2040), ylim=c(1, 5), type="l", col="red" )
par(new=TRUE)
plot( y$batch, log(y$time) ,xlim=c(0, 2040), ylim=c(1, 5), type="l", col="green" )
abline(h=3.84)