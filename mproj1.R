#Eric Olberding 10/16/2019
#LSTM neural nets vs ARIMA
#Simulated data is created using various simulation models
#These simulated datasets are analyzed using analysis models
#The forecasting accuracy of the various methods are compared for 1, 5, and 15 step ahead forecasts
#


#library(M4comp2018) not necessary now that we have data
library(forecast)
library(tensorflow)
library(keras)
library(ggplot2)
library(gridExtra)
set.seed(123)

###############################################################################
#Data Exploration to decide on various values in simulation study
###############################################################################



#explore industry time series
indM4 <- Filter(function(l) l$type == "Industry", M4)
length(indM4)
nind = rep(0, length(indM4))
for (i in 1:length(indM4)){
  nind[i] = indM4[[i]]$n 
}
summary(nind)
indM4[[1]]$pt_ff

#Standardize all of the time series in the industry category
sindM4 = indM4
for (i in 1:length(indM4)){
  sindM4[[i]]$x = (indM4[[i]]$x-mean(indM4[[i]]$x))/sqrt(var(indM4[[i]]$x))
}
nsind = rep(0, length(sindM4))
for (i in 1:length(sindM4)){
  nsind[i] = mean(sindM4[[i]]$x) 
}
summary(nsind)


#Creates list of arima(1,0,0) objects, uses try function to filter out time series
#that cannot be fit this way.
a1s = list(auto.arima(sindM4[[1]]$x, D=0, max.p=1, max.q=0, max.P=0, max.Q = 0, start.p = 1, seasonal = FALSE, stepwise = TRUE))

for(j in 2:length(sindM4)){

  tt = try(auto.arima(sindM4[[j]]$x, D=0, max.p=1, max.q=0, max.P=0, max.Q = 0, start.p = 1, seasonal = FALSE, stepwise = TRUE))
  
  
  if(!is(tt,"try-error")){
    a1 = auto.arima(sindM4[[j]]$x, D=0, max.p=1, max.q=0, max.P=0, max.Q = 0, start.p = 1, seasonal = FALSE, stepwise = TRUE)
    a1s[[length(a1s)+1]] = a1
  }
}


#order of the arima(p,q,d) models fit subject to the above constraints
ord = c(a1s[[1]]$arma)
for (i in 2:length(a1s)){
  ord = rbind(ord,  a1s[[i]]$arma)
}
summary(ord)


#find the subset of arima models with p=1, q=0
t = c(0)
for (i in 1:length(a1s)){
  t[i] = a1s[[i]]$arma[1] 
}
summary(t)
arma1 =a1s[which(t==1)]


#order of the arima models with p=1
ord1 = c(arma1[[1]]$arma)
for (i in 2:length(arma1)){
  ord1 = rbind(ord1,  arma1[[i]]$arma)
}
summary(ord1)


cfs = c(arma1[[1]]$coef[1], arma1[[1]]$sigma2)
cfs
for( i in 2:length(arma1)){
  cfs = rbind(cfs, c(arma1[[i]]$coef[1], arma1[[i]]$sigma2))
}

#6 number summary of the coefficient and variance
#we do this to decide what coefficients to use in simulated models
#This gives us what we want for the simple model
#used 3 quartiles of coefficient and mean of variance
cfsm = apply(cfs, 2, summary)
cfsm







##########################################################################
#Simulates time series for analysis
##########################################################################

#Simple time series (AR1) using quartiles of coefficients derived from above
#100 generated time series and each series has length 2000
#error terms have mean 3509 with variance 46328
#autoregressive coefficient is 0.95
s1 = list(0)
s2 = list(0)
s3 = list(0)

#first quartile ar1 coefficient 
for (i in 1:100){
  e = c(rnorm(2000, mean = 0, sd = sqrt(0.25))) 
  y = c(e[1])
  for (j in 2:2000){
    y[j] = -0.41927*y[j-1]+e[j]
  }
  s1[[i]] = y
}

s1ary = do.call(rbind, s1)

#split into training and test set (remember to go back and split into set for less and more 
#training data)
tests1 = t(apply(s1ary, 1, tail, n =200))

trains1 = t(apply(s1ary, 1, head, n =1800))

#median ar1 coefficient 
for (i in 1:100){
  e = c(rnorm(2000, mean = 0, sd = sqrt(0.25))) 
  y = c(e[1])
  for (j in 2:2000){
    y[j] = -0.22901*y[j-1]+e[j]
  }
  s2[[i]] = y
}


s2ary = do.call(rbind, s2)

#split into training and test set (remember to go back and split into set for less and more 
#training data)
tests2 = t(apply(s2ary, 1, tail, n =200))

trains2 = t(apply(s2ary, 1, head, n =1800))




#3rd quartile ar1 coefficient 
for (i in 1:100){
  e = c(rnorm(2000, mean = 0, sd = sqrt(0.25))) 
  y = c(e[1])
  for (j in 2:2000){
    y[j] = 0.36374*y[j-1]+e[j]
  }
  s3[[i]] = y
}

s3ary = do.call(rbind, s3)

#split into training and test set (remember to go back and split into set for less and more 
#training data)
tests3 = t(apply(s3ary, 1, tail, n =200))

trains3 = t(apply(s3ary, 1, head, n =1800))


#coefficients are the same as above except for the coefficients on the autoregressive components
#In this version we make the coefficients satisfy the stationarity constraints
#coefficient for linear component is 0.7 and the nonlinear component has coefficient 0.28
#I CAN"T USE THE SAME COEFFICIENTs BECAUSE IT BLOWS UP

#nonlinear time series 1st quartile
n1 = list(0)

for (i in 1:100){
  e = c(rnorm(2000, mean = 0, sd = sqrt(0.25)))
  y = c(e[1])
  for (j in 2:2000){
    y[j] = -0.41927*(sign(y[j-1])*abs(y[j-1])^1.2)+e[j]
  }
  n1[[i]] = y
}
n1ary = do.call(rbind, n1)



#checks for numerical errors (infinity and NaN); there are none
tot = 0
for ( i in 1:length(n1)){
  tot = tot + sum(is.infinite(n1[[i]]))
}
tot

tot = 0
for ( i in 1:length(n1)){
  tot = tot + sum(is.nan(n1[[i]]))
}
tot

#split into training and test set (remember to go back and split into set for less and more 
#training data)
testn1 = t(apply(n1ary, 1, tail, n =200))

trainn1 = t(apply(n1ary, 1, head, n =1800))

#nonlinear time series median
n2 = list(0)

for (i in 1:100){
  e = c(rnorm(2000, mean = 0, sd = sqrt(0.25)))
  y = c(e[1])
  for (j in 2:2000){
    y[j] =-0.22901*(sign(y[j-1])*abs(y[j-1])^1.2)+e[j]
  }
  n2[[i]] = y
}
n2ary = do.call(rbind, n2)
n2


#checks for numerical errors (infinity and NaN) there are none
totinf = 0
for ( i in 1:length(n2)){
  totinf = totinf + sum(is.infinite(n2[[i]]))
}
totinf

totnan = 0
for ( i in 1:length(n2)){
  totnan = totnan + sum(is.nan(n2[[i]]))
}
totnan


#split into training and test set (remember to go back and split into set for less and more 
#training data)
testn2 = t(apply(n2ary, 1, tail, n =200))

trainn2 = t(apply(n2ary, 1, head, n =1800))

#nonlinear time series 3rd quartile
n3 = list(0)

for (i in 1:100){
  e = c(rnorm(2000, mean = 0, sd = sqrt(0.25)))
  y = c(e[1])
  for (j in 2:2000){
    y[j] = 0.36374*(sign(y[j-1])*abs(y[j-1])^1.2)+e[j]
  }
  n3[[i]] = y
}
n3ary = do.call(rbind, n3)
n3


#checks for numerical errors (infinity and NaN) there are none
totinf = 0
for ( i in 1:length(n3)){
  totinf = totinf + sum(is.infinite(n3[[i]]))
}
totinf

totnan = 0
for ( i in 1:length(n3)){
  totnan = totnan + sum(is.nan(n3[[i]]))
}
totnan


#split into training and test set (remember to go back and split into set for less and more 
#training data)
testn3 = t(apply(n3ary, 1, tail, n =200))

trainn3 = t(apply(n3ary, 1, head, n =1800))


#this can be used to check that the generated training data is nonseasonal. All generated
#datasets are nonseasonal.
seasonality = list()
for (i in 1:100){
  s=ets(trainn3[i,])
  seasonality[[i]]=s$components[3]
}
seasonality


#######################################################################
#######################################################################
#Fit Analysis Models
#######################################################################
#######################################################################


#get MASE for each generated time series and associated forecasts
#(only works with output from forecast function on auto.arima from Forecast package)
getm = function(dat){
  m = c()
  
  for (i in 1:length(dat)){
    m[i] = dat[[i]][2,6]
  }

  m
}


#Returns a list, the outer list iterates over forecast horizons. The inner list iterates over the mean mase, a vector of MASE for
#each generated series, and a vector of forecasts for each series a certain length
frcsts = function(models, horizons, testdat){
  forecasts = list()
  for(i in 1:length(horizons)){
    #set of forecasts for a given horizon and list of models
    forecasts[[i]]= lapply(models, forecast, h=horizons[i])
    
    #MASE statistic for a set of forecasts
    mase = getm(lapply(forecasts[[i]], accuracy, x=testdat[1:horizons[i]]))
    
    forecasts[[i]]= list(mean(mase),mase)
  }
  forecasts
}

#gets the std dev of the mase for arima models
masestd = function(forecasts, horizons){
  sdev = c()
  for(i in 1:length(horizons)){
    sdev[i]=sd(forecasts[[i]][[2]])
  }
  sdev
}

#this plots two different sets of mases against each other. The x axis is the set of forecast horizons.
maseplots = function(mas1,mas2,horizons, p, title='title'){
  
  #get the averaged mase for each forecast horizon as a vector
  mase1=c()
  mase2=c()
  mase1upper=c()
  mase1lower=c()
  mase2upper=c()
  mase2lower=c()
  for(i in 1:length(horizons)){
    mase1[i]=mas1[[i]][[1]]
    mase2[i]=mas2[[i]][[1]]
    mase1upper[i]=quantile(mas1[[i]][[2]], probs = p[2])
    mase1lower[i]=quantile(mas1[[i]][[2]], probs = p[1])
    mase2upper[i]=quantile(mas2[[i]][[2]], probs = p[2])
    mase2lower[i]=quantile(mas2[[i]][[2]], probs = p[1])
  }
  dataframe = data.frame(cbind(horizons, mase1, mase1upper, mase1lower, 
                               mase2, mase2upper, mase2lower))
  
  p = (ggplot(dataframe, aes(horizons)) + 
    geom_line(aes(y = mase1, colour = "LSTM")) + 
    geom_line(aes(y = mase2, colour = "ARIMA"))+ 
    geom_line(linetype='dashed',aes(y = mase1upper, colour = 'LSTM'))+
    geom_line(linetype='dashed',aes(y = mase1lower, colour = 'LSTM'))+
    geom_line(linetype='dashed',aes(y = mase2upper, colour = 'ARIMA'))+
    geom_line(linetype='dashed',aes(y = mase2lower, colour = 'ARIMA'))+
      theme(legend.position = "none")+
      labs(y= "MASE", x = "horizon")+ggtitle(title) )
  
}


#forecast horizons
#h=c(1,5,10,100)
h=1:100



####################
#Auto.arima function
####################



#for simple case 1st quartile
aas1 = apply(trains1, 1, auto.arima)

fs1 = frcsts(aas1, horizons = h, tests1)

sdfs1=masestd(fs1,h)

#short version

shortaas1 = apply(trains1[,1600:1800], 1, auto.arima)

shortfs1 = frcsts(shortaas1, horizons = h, tests1)

sdshortfs1=masestd(shortfs1,h)

#######################
#for simple case median
#######################

aas2 = apply(trains2, 1, auto.arima)

fs2 = frcsts(aas2, horizons = h, tests2)

sdfs2=masestd(fs2,h)

#short version

shortaas2 = apply(trains2[,1600:1800], 1, auto.arima)

shortfs2 = frcsts(shortaas2, horizons = h, tests2)

sdshortfs2 = masestd(shortfs2,h)

########################
#for simple 3rd quartile
########################

aas3 = apply(trains3, 1, auto.arima)

fs3 = frcsts(aas3, horizons = h, tests3)

sdfs3=masestd(fs3,h)

#short version

shortaas3 = apply(trains3[,1600:1800], 1, auto.arima)

shortfs3 = frcsts(shortaas3, horizons = h, tests3)

sdshortfs3=masestd(shortfs3,h)

################################
#for nonlinear case 1st quartile
################################

aan1 = apply(trainn1, 1, auto.arima)
fn1 = frcsts(aan1, horizons = h, testn1)

sdfn1=masestd(fn1,h)

#short version

shortaan1 = apply(trainn1[,1600:1800], 1, auto.arima)

shortfn1 = frcsts(shortaan1, horizons = h, testn1)

sdshortfn1=masestd(shortfn1,h)

#########################
#for nonlinear case median
######################### 
aan2 = apply(trainn2, 1, auto.arima)

fn2 = frcsts(aan2, horizons = h, testn2)

sdfn2=masestd(fn2,h)

#short version

shortaan2 = apply(trainn2[,1600:1800], 1, auto.arima)

shortfn2 = frcsts(shortaan2, horizons = h, testn2)

sdshortfn2=masestd(shortfn2,h)

################################
#for nonlinear case 3rd quartile
################################
aan3 = apply(trainn3, 1, auto.arima)
fn3 = frcsts(aan3, horizons = h, testn3)

sdfn3=masestd(fn3,h)

#short version

shortaan3 = apply(trainn3[,1600:1800], 1, auto.arima)

shortfn3 = frcsts(shortaan3, horizons = h, testn3)

sdshortfn3=masestd(shortfn3,h)






#########################################################
#########################################################
#LSTM models     #my computer correctly ran an example from the internet so hopefully no software version issues
#########################################################
#########################################################


#general model


#change dim of data to be appropriate for LSTM keras. Note tat we assume the number of features at each time step is 1 (last dimension)
reshape_X_3d <- function(X) {
  dim(X) <- c(nrow(X), ncol(X), 1)
  X
}

FLAGS <- flags(
  # There is a so-called "stateful LSTM" in Keras. While LSTM is stateful
  # per se, this adds a further tweak where the hidden states get 
  # initialized with values from the item at same position in the previous
  # batch. This is helpful just under specific circumstances, or if you want
  # to create an "infinite stream" of states, in which case you'd use 1 as 
  # the batch size. Below, we show how the code would have to be changed to
  # use this, but it won't be further discussed here.
  flag_boolean("stateful", FALSE),
  # number of samples fed to the model in one go
  flag_integer("batch_size", 200),
  # size of the hidden state, equals size of predictions
  flag_integer("n_timesteps", 50),
  # how many epochs to train for
  flag_integer("n_epochs", 50),
  # fraction of the units to drop for the linear transformation of the inputs
  flag_numeric("dropout", 0),
  # fraction of the units to drop for the linear transformation of the 
  # recurrent state
  flag_numeric("recurrent_dropout", 0),
  # loss function. mean squared error
  flag_string("loss", "MSE"),
  # optimizer = stochastic gradient descent. Seemed to work better than adam 
  # or rmsprop here (as indicated by limited testing)
  flag_string("optimizer_type", "sgd"),
  # size of the LSTM layer
  flag_integer("n_units", 5),
  # learning rate
  flag_numeric("lr", 0.6),
  # momentum, an additional parameter to the SGD optimizer
  flag_numeric("momentum", 0),
  # parameter to the early stopping callback
  flag_integer("patience", 30)
)


#this function formats the data for LSTM input where each sample(row) is a time series of length samplen.
#from trnstrt to trnend must be divisible by samplen (length of sample)
#samplen must be equal to n_timesteps in the FLAGS set
formdat = function(dat,trnstrt,trnend, samplen){
  ss = list()
  
  if (is.array(dat)==TRUE){
    for (i in (trnstrt+samplen):(trnend+1)) {
      ss[[i+1-(trnstrt+samplen)]] = dat[,(i-samplen):i]
    }
    ss = do.call(rbind, ss)
    ts = list()
    for (i in (trnend+2+samplen):ncol(dat)){
      ts[[i+1-(trnend+2+samplen)]] = dat[,(i-samplen):i]
    }
  }
  
  else{
    for (i in (trnstrt+samplen):(trnend+1)) {
      ss[[i+1-(trnstrt+samplen)]] = dat[(i-samplen):i]
    }
    ss = do.call(rbind, ss)
    ts = list()
    for (i in (trnend+2+samplen):length(dat)){
      ts[[i+1-(trnend+2+samplen)]] = dat[(i-samplen):i]
    }
  }
  ts=do.call(rbind,ts)
  out=rbind(ss,ts)
  out
}


#trnend+1-trnstrt must be divisible by samplen for this to work 
fitlstm = function(dat,flgs,center=0,scl=1,trnstrt,trnend, samplen){
  
  
  # how many features/predictors we have
  n_features <- 1
  # just in case we wanted to try different optimizers, we could add here
  optimizer <- switch(flgs$optimizer_type,
                      sgd = optimizer_sgd(lr = flgs$lr, 
                                          momentum = flgs$momentum)
  )
  
  
  # callbacks to be passed to the fit() function
  # We just use one here: we may stop before n_epochs if the loss on the
  # validation set does not decrease (by a configurable amount, over a 
  # configurable time)
  clbacks <- list(
    callback_early_stopping(patience = flgs$patience)
  )
  
  # create the model
  model <- keras_model_sequential()
  
  # add the LSTM layer
  model %>%
    layer_lstm(
      units = flgs$n_units, 
      # the first layer in a model needs to know the shape of the input data
      input_shape  = c(flgs$n_timesteps, n_features),
      dropout = flgs$dropout,
      recurrent_dropout = flgs$recurrent_dropout,
      # by default, an LSTM just returns the final state
      return_sequences = FALSE
    )%>%
    layer_dense(units = 1)
  model %>%
    compile(loss = flgs$loss,
            optimizer = optimizer)
 
  if(is.array(dat)==TRUE){ 
    #fit the model to the data
    trnrows = (trnend+2-trnstrt-samplen)*nrow(dat)
    formd = formdat(dat,trnstrt,trnend, samplen)
    m <- model %>% fit(
      x          = reshape_X_3d((formd[1:trnrows,1:samplen]-center)/scl),
      y          = (formd[1:trnrows,(samplen+1)]-center)/scl,
      validation_data = list(reshape_X_3d((formd[(trnrows+1):nrow(formd),1:samplen]-center)/scl), (formd[(trnrows+1):nrow(formd),(samplen+1)]-center)/scl),
      batch_size = flgs$batch_size,
      epochs     = flgs$n_epochs,
      callbacks = clbacks
    )
  }
  
  else{
    #fit the model to the data
    trnrows = (trnend+2-trnstrt-samplen)
    formd = formdat(dat,trnstrt,trnend, samplen)
    m <- model %>% fit(
      x          = reshape_X_3d((formd[1:trnrows,1:samplen]-center)/scl),
      y          = (formd[1:trnrows,(samplen+1)]-center)/scl,
      validation_data = list(reshape_X_3d((formd[(trnrows+1):nrow(formd),1:samplen]-center)/scl), (formd[(trnrows+1):nrow(formd),(samplen+1)]-center)/scl),
      batch_size = flgs$batch_size,
      epochs     = flgs$n_epochs,
      callbacks = clbacks
    )
  }
  m
}

#returns predictions for a particular LSTM model and dataset. Starts with some input data
#of length L, generates predictions based upon this input. Removes first time step from
#data and add most recent prediction to the end. This new input is fed in to make the 
#next prediction.
#output is an array where each row is of length numpreds (the forecasts for sample 1 is in row 1)
prdvec = function(dat,modname, numpreds, cnt, scl){
  
  model = load_model_hdf5(modname)
  
  inpt = dat
  pred = list()
  for(i in 1:numpreds){
    pred[[i]] <- predict(model, reshape_X_3d((inpt[,1:ncol(inpt)]-cnt)/scl), batch_size = 1)
    inpt = cbind(inpt[,2:ncol(inpt)],(pred[[i]]*scl+cnt))
    print(i)
    flush.console()
  }
  pred = t(do.call("cbind",(pred)))
  pred = pred*scl+cnt
  pred
}

#this function  computes the MASE for the LSTM predictions
#test data and predictions must be the same length
#returns the MASE for each sample series
foracc = function(traindat, testdat, preds){
  
  denom = apply(abs(traindat[,2:ncol(traindat)]-traindat[,1:(ncol(traindat)-1)]),1,sum)/(ncol(traindat)-1)
  
  if(is.null(ncol(preds))){
    numerators = abs(preds-testdat) 
  }
  else{
    numerators = apply(abs(preds-testdat),1,mean)
  }
  mase = numerators/denom
}

lstmfrcsts = function(traindat, testdat, prds, horizons){
  
  mase = list()
  for(i in 1:length(horizons)){
    mase.per.series = foracc(traindat, testdat[,1:horizons[i]],prds[,1:horizons[i]])
    mase[[i]] = list(mean(mase.per.series),mase.per.series)
  }
  mase
}

mae = function(testdat,prds, horizon){
  mae=list()
  for(i in 1:length(horizons)){
    a = abs(testdat[,1:horizons[i]]-prds[,1:horizons[i]])
    mae[[i]]=list(mean(a))
  }
}

#############################
#for simple case 1st quartile
#############################

lstms1 = apply(trains1, 1, fitlstm, FLAGS, 0, 10, trnstrt = 1, trnend = 1600, samplen = 50)
?apply
model %>% save_model_hdf5('lstms1')

prds1 = prdvec(trains1[,1751:1800],'lstms1',100, cnt = 0, scl = 10)

lms1 = lstmfrcsts(trains1, tests1,prds1,horizons = h)

sdlms1 = masestd(lms1,h)

a = abs(trains1[,1:h[length(h)]]-prds1[,1:h[length(h)]])

#short dataset

shortlstms1 = fitlstm(trains1, FLAGS, 0, 10, trnstrt = 1601, trnend = 1700, samplen = 50)

model %>% save_model_hdf5('shortlstms1')

shortprds1 = prdvec(trains1[,1751:1800],'shortlstms1',100, cnt = 0, scl = 10)

shortlms1 = lstmfrcsts(trains1[,1601:1800], tests1,shortprds1,horizons = h)

sdshortlms1 = masestd(shortlms1,h)

#######################
#for simple case median
#######################


lstms2 = fitlstm(trains2, FLAGS, 0, 10, trnstrt = 1, trnend = 1600, samplen = 200)

model %>% save_model_hdf5('lstms2')

prds2 = prdvec(trains2[,1601:1800],'lstms2',100, cnt = 0, scl = 10)

lms2 = lstmfrcsts(trains2, tests2,prds2,horizons = h)

sdlms2 = masestd(lms2,h)

#short dataset

shortlstms2 = fitlstm(trains2, FLAGS, 0, 10, trnstrt = 1601, trnend = 1700, samplen = 50)

model %>% save_model_hdf5('shortlstms2')

shortprds2 = prdvec(trains2[,1751:1800],'shortlstms2',100, cnt = 0, scl = 10)

shortlms2 = lstmfrcsts(trains2[,1601:1800], tests2,shortprds2,horizons = h)

sdshortlms2 = masestd(shortlms2,h)

##############################
#for simple case 3rd quartiles
##############################


lstms3 = fitlstm(trains3, FLAGS, 0, 10, trnstrt = 1, trnend = 1600, samplen = 200)

model %>% save_model_hdf5('lstms3')

prds3 = prdvec(trains3[,1601:1800],'lstms3',100, cnt = 0, scl = 10)

lms3 = lstmfrcsts(trains3, tests3,prds3,horizons = h)

sdlms3 = masestd(lms3,h)

#short dataset

shortlstms3 = fitlstm(trains3, FLAGS, 0, 10, trnstrt = 1601, trnend = 1700, samplen =50)

model %>% save_model_hdf5('shortlstms3')

shortprds3 = prdvec(trains3[,1751:1800],'shortlstms3',100, cnt = 0, scl = 10)

shortlms3 = lstmfrcsts(trains3[,1601:1800], tests3,shortprds3,horizons = h)

sdshortlms3 = masestd(shortlms3,h)

################################
#for nonlinear case 1st quartile
################################


lstmn1 = fitlstm(trainn1, FLAGS, 0, 10, trnstrt = 1, trnend = 1600, samplen = 200)

model %>% save_model_hdf5('lstmn1')

prdn1 = prdvec(trainn1[,1601:1800],'lstmn1',100, cnt = 0, scl = 10)

lmn1 = lstmfrcsts(trainn1, testn1,prdn1,horizons = h)

sdlmn1 = masestd(lmn1,h)

#short dataset

shortlstmn1 = fitlstm(trainn1, FLAGS, 0, 10, trnstrt = 1601, trnend = 1700, samplen = 50)

model %>% save_model_hdf5('shortlstmn1')

shortprdn1 = prdvec(trainn1[,1751:1800],'shortlstmn1',100, cnt = 0, scl = 10)

shortlmn1 = lstmfrcsts(trainn1[,1601:1800], testn1,shortprdn1,horizons = h)

sdshortlmn1 = masestd(shortlmn1,h)

##########################
#for nonlinear case median
##########################

lstmn2 = fitlstm(trainn2, FLAGS, 0, 10, trnstrt = 1, trnend = 1600, samplen = 200)

model %>% save_model_hdf5('lstmn2')

prdn2 = prdvec(trainn2[,1601:1800],'lstmn2',100, cnt = 0, scl = 10)

lmn2 = lstmfrcsts(trainn2, testn2,prdn2,horizons = h)

sdlmn2 = masestd(lmn2,h)
  
#short dataset

shortlstmn2 = fitlstm(trainn2, FLAGS, 0, 10, trnstrt = 1601, trnend = 1700, samplen = 50)

model %>% save_model_hdf5('shortlstmn2')

shortprdn2 = prdvec(trainn2[,1751:1800],'shortlstmn2',100, cnt = 0, scl = 10)

shortlmn2 = lstmfrcsts(trainn2[,1601:1800], testn2,shortprdn2,horizons = h)

sdshortlmn2 = masestd(shortlmn2,h)

#################################
#for nonlinear case 3rd quartiles
#################################

lstmn3 = fitlstm(trainn3, FLAGS, 0, 10, trnstrt = 1, trnend = 1600, samplen = 200)

model %>% save_model_hdf5('lstmn3')

prdn3 = prdvec(trainn3[,1601:1800],'lstmn3',100, cnt = 0, scl = 10)

lmn3 = lstmfrcsts(trainn3, testn3,prdn3,horizons = h)

sdlmn3 = masestd(lmn3,h)

#short dataset

shortlstmn3 = fitlstm(trainn3, FLAGS, 0, 10, trnstrt = 1601, trnend = 1700, samplen = 50)

model %>% save_model_hdf5('shortlstmn3')

shortprdn3 = prdvec(trainn3[,1751:1800],'shortlstmn3',100, cnt = 0, scl = 10)

shortlmn3 = lstmfrcsts(trainn3[,1601:1800], testn3,shortprdn3,horizons = h)

sdshortlmn3 = masestd(shortlmn3,h)

sdhtable = function(i){
  lin1 = c(sdshortlms1[i],sdshortfs1[i],sdlms1[i],sdfs1[i])
  lin2 = c(sdshortlms2[i],sdshortfs2[i],sdlms2[i],sdfs2[i])
  lin3 = c(sdshortlms3[i],sdshortfs3[i],sdlms3[i],sdfs3[i])
  nlin1 = c(sdshortlmn1[i],sdshortfn1[i],sdlmn1[i],sdfn1[i])
  nlin2 = c(sdshortlmn2[i],sdshortfn2[i],sdlmn2[i],sdfn2[i])
  nlin3 = c(sdshortlmn3[i],sdshortfn3[i],sdlmn3[i],sdfn3[i])
  
  htable = rbind(lin1,lin2,lin3,nlin1,nlin2,nlin3)
  htable
  
}
sdhtable(1)
sdhtable(2)
sdhtable(3)
sdhtable(4)

p = c(0.05,0.95)
#generates plots for models fit on long dataset
ps1=maseplots(lms1, fs1,h, p, title='Linear 1')
ps2=maseplots(lms2, fs2,h, p, title='Linear 2')
ps3=maseplots(lms3, fs3,h, p, title='Linear 3')

pn1=maseplots(lmn1, fn1,h, p, title='Non-linear 1')
pn2=maseplots(lmn2, fn2,h, p, title='Non-linear 2')
pn3=maseplots(lmn3, fn3,h, p, title='Non-linear 3')

gs=arrangeGrob(ps1,ps2,ps3,pn1,pn2,pn3, nrow=2, top = 'Large Sample Models')
grid.arrange(gs)

#generates plots for models fit on long dataset
sps1=maseplots(shortlms1, shortfs1,h, p, title='Linear 1')
sps2=maseplots(shortlms2, shortfs2,h, p, title='Linear 2')
sps3=maseplots(shortlms3, shortfs3,h, p, title='Linear 3')

spn1=maseplots(shortlmn1, shortfn1,h, p, title='Non-linear 1')
spn2=maseplots(shortlmn2, shortfn2,h, p, title='Non-linear 2')
spn3=maseplots(shortlmn3, shortfn3,h, p, title='Non-linear 3')

sgs=arrangeGrob(sps1,sps2,sps3,spn1,spn2,spn3, nrow=2, top='Small Sample Models')
grid.arrange(sgs)


#check of model structure: given 1 through 100, predict next time step. Finally got it to work!!!
a = seq(from =1, to =100)
anorm = (a-mean(a))/500

ar =  list()

for (i in 4:100) {
  ar[[i-3]] = anorm[(i-3):i]
}
ar = do.call(rbind, ar)
ar

for (i in 1:2000) {
  check <- model %>% fit(
    x          = reshape_X_3d(ar[1:86,]),
    y          = anorm[5:90],
    validation_data = list(reshape_X_3d(ar[91:96,]), anorm[95:100]),
    batch_size = 5,
    epochs     = 1,
    callbacks = callbacks
  )
  reset_states(model)
}
pred_train <- model %>%
  predict(reshape_X_3d(ar[97,]), batch_size = 1) %>%
  .[, 1]
ar[,1]
pred_train*500+mean(a)
trains1[1,1:1200]
summary(model)

#[[1]] is set of weights for input [[2]] is for hidden state [[3]] is for bias
#each row is the corresponding weight for input forget cell state output (so 5 nodes means 5 input weights then 5 forget weights etc)
#after checking weights we see that they are nonzero
get_weights(model)
get_weights(get_layer(model,index=1))

##################################################################
#Not currently used for the project
##################################################################
#explore Finance time series
finM4 <- Filter(function(l) l$type == "Finance", M4)
length(finM4)
nfin = rep(0, length(finM4))
for (i in 1:length(finM4)){
  nfin[i] = finM4[[i]]$n 
}
summary(nfin)



#explore Demographic time series
demM4 <- Filter(function(l) l$type == "Demographic", M4)
length(demM4)
ndem = rep(0, length(demM4))
for (i in 1:length(demM4)){
  ndem[i] = demM4[[i]]$n 
}
summary(ndem)



#explore Macro time series
macM4 <- Filter(function(l) l$type == "Macro", M4)
length(macM4)
nmac = rep(0, length(macM4))
for (i in 1:length(macM4)){
  nmac[i] = macM4[[i]]$n 
}
summary(nmac)


#explore Micro time series
micM4 <- Filter(function(l) l$type == "Micro", M4)
length(micM4)
nmic = rep(0, length(micM4))
for (i in 1:length(micM4)){
  nmic[i] = micM4[[i]]$n 
}
summary(nmic)


summary(model)
#explore Other time series
otM4 <- Filter(function(l) l$type == "Other", M4)
length(otM4)
not = rep(0, length(otM4))
for (i in 1:length(otM4)){
  not[i] = otM4[[i]]$n 
}
summary(not)
