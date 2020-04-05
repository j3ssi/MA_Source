#fitte geraden

#s=3a

count3 <- c(100858, 198074, 295290, 392506, 489722)
batch3 <- c(18656, 12288, 8000, 5952, 4736)
coeff  <- c(5.40, 8.06, 12.30, 16.49, 20.68)
size <- c(1,2,3,4,5)

gerade3<-lm(coeff ~ size)
summary(gerade3)

