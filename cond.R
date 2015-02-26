require(cond)

theta <- 1.5
lambda <- 2
n.max <- 500

n.vals <- 5:n.max
N <- length(n.vals)

theta.est <- array(dim=N)
theta.est.cond <- array(dim=N)

sigma_n <- function(n) {
        0.2 * log(n + 5) ** 0.4
        #1
        #log(log(n + 10))
}

for(i in 1:N) {
        n <- n.vals[i]
        
        sigma <- sigma_n(n)

        while(TRUE) {
                x <- rnorm(n,0,sigma)
        
                logq <- (theta * x + lambda
                                 - log(n) - (theta * sigma) ** 2 / 2)
                q <- exp(logq)
                p <- q / (1 + q)
        
                y <- runif(n) < p
                
                # Owen and Roediger (2014)
                n.1 <- sum(y)
                n.0 <- n - n.1
                l.0 <- min(x[y == 0])
                u.0 <- max(x[y == 0])
                l.1 <- min(x[y == 1])
                u.1 <- max(x[y == 1])
                if (n.0 == 0 || n.1 == n || l.0 >= u.1 || l.1 >= u.0) next

                d <- data.frame(y=y, x=x)
                fm <- glm(y ~ x, data=d, family=binomial(link="logit"))
                fm.theta.est <- fm$coef["x"]
                if (-5 < fm.theta.est && fm.theta.est < 5) break
        }
                
        fm.cond <- cond(fm, x)
        print(summary(fm.cond))
        theta.est[i] <- fm.theta.est
        theta.est.cond[i] <- fm.cond$coef[2,1]
}

