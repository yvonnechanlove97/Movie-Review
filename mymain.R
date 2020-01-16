rm(list=ls())
library(text2vec)
library(magrittr)
library(glmnet)
#get data
all = read.table("data.tsv",stringsAsFactors = F,header = T)
all$review = gsub('<.*?>', ' ', all$review)
splits = read.table("splits.csv", header = T)
s = 3  # Here we get the 3rd training/test split. 
myvocab=as.matrix(read.table("myVocab.txt"))

train = all[-which(all$new_id%in%splits[,s]),]
test = all[which(all$new_id%in%splits[,s]),]
#build the vocabulary
prep_fun = tolower
tok_fun = word_tokenizer

train_tokens = train$review %>% 
  prep_fun %>% 
  tok_fun
it_train = itoken(train_tokens, 
                  ids = train$new_id,
                  progressbar = FALSE)
it_test = test$review %>% 
  prep_fun %>% tok_fun %>% 
  itoken(ids = test$new_id, progressbar = FALSE)

stop_words2 = c('a','about',
                'above','after','again','against','ain','all','am','an','and','any',
                'are','aren',"aren't",'as','at','be','because','been','before',
                'being','below','between','both','but','by','can','couldn',"couldn't",
                'd','did','didn',"didn't",'do','does','doesn',"doesn't",
                'doing','don',"don't",'down','during','each','few','for','from',
                'further','had','hadn',"hadn't",'has','hasn',"hasn't",
                'have','haven',"haven't",'having','he','her','here','hers',
                'herself','him','himself','his','how','i','if','in','into','is',
                'isn',"isn't",'it',"it's",'its','itself','just','ll','m','ma',
                'me','mightn',"mightn't",'more','most','mustn',"mustn't",
                'my','myself','needn',"needn't",'no','nor','not','now',
                'o','of','off','on','once','only','or','other','our',
                'ours','ourselves','out','over','own','re','s',
                'same','shan',"shan't",'she',"she's",
                'should',"should've",'shouldn',"shouldn't",'so','some',
                'such','t','than','that',"that'll",'the','their',
                'theirs','them','themselves','then','there','these',
                'they','this','those','through','to','too','under',
                'until','up','ve','very','was','wasn',"wasn't",'we','were',
                'weren',"weren't",'what','when','where','which',
                'while','who','whom','why','will','with','won',
                "won't",'wouldn',"wouldn't",'y','you',"you'd","you'll",
                "you're","you've",'your','yours','yourself','yourselves'
)

vocab=create_vocabulary(it_train,ngram = c(1L,4L),stopwords = stop_words2)

#clean train vocab
pruned_vocab = prune_vocabulary(vocab, 
                                term_count_min = 5, 
                                doc_proportion_max = 0.5,
                                doc_proportion_min = 0.001)


vectorizer = vocab_vectorizer(pruned_vocab)
dtm_train  = create_dtm(it_train, vectorizer)
dtm_test = create_dtm(it_test, vectorizer)

train_x=dtm_train[,which(colnames(dtm_train)%in%myvocab)]
test_x=dtm_test[,which(colnames(dtm_test)%in%myvocab)]


#ridge
NFOLDS = 10
mycv = cv.glmnet(x=train_x, y=train$sentiment, 
                 family='binomial',type.measure = "auc", 
                 nfolds = NFOLDS, alpha=0)
myfit = glmnet(x=train_x, y=train$sentiment, 
               lambda = mycv$lambda.min, family='binomial', alpha=0)
logit_pred = predict(myfit, test_x, type = "response")
glmnet:::auc(test$sentiment, logit_pred)
write.table(result,"mysubmission.txt",row.names=FALSE, col.names = c('new_id','prob'), sep=", ")

#result=cbind(test$new_id,logit_pred)
#write.table(result,"Result_3.txt",row.names=FALSE, col.names = c('new_id','prob'), sep=", ")
#final_vocab2=words[id]
#write.table(final_vocab2,"myVocab1.txt",row.names=FALSE, col.names=FALSE,sep=", ")

#three
#3 0.9609737
#2 0.9635941
#1 0.9643016