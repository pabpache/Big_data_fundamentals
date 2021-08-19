#Name: Pablo Pacheco
#znumber: z5222810
from pyspark.sql import *
from pyspark.sql import DataFrame
from pyspark.ml.classification import LogisticRegression, LinearSVC, NaiveBayes

from pyspark.sql.types import DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, CountVectorizer, StringIndexer
from pyspark.sql.functions import udf

def base_features_gen_pipeline(input_descript_col="descript", input_category_col="category", output_feature_col="features", output_label_col="label"):
    #Doing the tokenization
    wordTokenizer= Tokenizer(inputCol=input_descript_col, outputCol='words')
    #Doing the bag of words
    bag_words= CountVectorizer(inputCol='words',outputCol=output_feature_col)
    
    #coverting into labels
    labels= StringIndexer(inputCol=input_category_col, outputCol= output_label_col)
    
    #building the pipeline
    genPipeline = Pipeline(stages=[wordTokenizer,bag_words,labels])
    
    return genPipeline

def gen_meta_features(training_df, nb_0, nb_1, nb_2, svm_0, svm_1, svm_2):
    
    #cross validation
    
    k=5         #number of groups
    for i in range(k):
        condition = training_df['group'] == i
        c_train = training_df.filter(~condition).cache()
        c_test = training_df.filter(condition).cache()
        
        nb_0_model= nb_0.fit(c_train)
        nb_0_predict= nb_0_model.transform(c_test)
        
        nb_1_model= nb_1.fit(c_train)
        nb_1_predict= nb_1_model.transform(c_test)
        #select only id and nb_pred_1 columns
        aux_df=nb_1_predict.select(nb_1_predict['id'],nb_1_predict['nb_pred_1'])
        #join both dataframes
        curr_df= nb_0_predict.join(aux_df,on=['id'])
        
        nb_2_model= nb_2.fit(c_train)
        nb_2_predict= nb_2_model.transform(c_test)
        aux_df=nb_2_predict.select(nb_2_predict['id'],nb_2_predict['nb_pred_2'])
        curr_df= curr_df.join(aux_df,on=['id'])
        
        svm_0_model= svm_0.fit(c_train)
        svm_0_predict= svm_0_model.transform(c_test)
        aux_df=svm_0_predict.select(svm_0_predict['id'],svm_0_predict['svm_pred_0'])
        curr_df= curr_df.join(aux_df,on=['id'])
        
        svm_1_model= svm_1.fit(c_train)
        svm_1_predict= svm_1_model.transform(c_test)
        aux_df=svm_1_predict.select(svm_1_predict['id'],svm_1_predict['svm_pred_1'])
        curr_df= curr_df.join(aux_df,on=['id'])
        
        svm_2_model= svm_2.fit(c_train)
        svm_2_predict= svm_2_model.transform(c_test)
        aux_df=svm_2_predict.select(svm_2_predict['id'],svm_2_predict['svm_pred_2'])
        curr_df= curr_df.join(aux_df,on=['id'])
        
        #the different k groups are disjoint, so if we union the dataframes, we can be sure that we are not repeating rows
        #Do union between different iterations
        
        if i == 0:
            main_df=curr_df
        else:
            main_df= main_df.union(curr_df)
            
    # function to create joint predictions 
    def joint_prediction(clf1,clf2):
        if clf1==0:
            if clf2==0:
                return float(0)
            else:
                return float(1)
        else:
            if clf2==0:
                return float(2)
            else:
                return float(3)
                
    j_predic= udf(joint_prediction, DoubleType())
    
    #add joint_prediction columns
    main_df= main_df.withColumn('joint_pred_0',j_predic('nb_pred_0','svm_pred_0'))
    main_df= main_df.withColumn('joint_pred_1',j_predic('nb_pred_1','svm_pred_1'))
    main_df= main_df.withColumn('joint_pred_2',j_predic('nb_pred_2','svm_pred_2'))            
             
    #select asked columns
    main_df= main_df.select(main_df['id'],main_df['group'],main_df['features'],main_df['label'],main_df['label_0'],
                            main_df['label_1'],main_df['label_2'],main_df['nb_pred_0'],main_df['nb_pred_1'],
                            main_df['nb_pred_2'],main_df['svm_pred_0'],main_df['svm_pred_1'],main_df['svm_pred_2'],
                            main_df['joint_pred_0'],main_df['joint_pred_1'],main_df['joint_pred_2'])
    
    return main_df
        

def test_prediction(test_df, base_features_pipeline_model, gen_base_pred_pipeline_model, gen_meta_feature_pipeline_model, meta_classifier):
     
    #preparing testing data
    trans_test_df= base_features_pipeline_model.transform(test_df)
    
    #generate new meta-features
    #use the base classifier which learnt from the whole data on the transformed test data
    new_predict= gen_base_pred_pipeline_model.transform(trans_test_df)
    ##print('new_predict_SCHEMA:')
    ##new_predict.printSchema()
    
    #add joint_predictions:
        # function to create joint predictions 
    def joint_prediction(clf1,clf2):
        if clf1==0:
            if clf2==0:
                return float(0)
            else:
                return float(1)
        else:
            if clf2==0:
                return float(2)
            else:
                return float(3)
                
    j_predic= udf(joint_prediction, DoubleType())
    
    #add joint_prediction columns
    new_predict= new_predict.withColumn('joint_pred_0',j_predic('nb_pred_0','svm_pred_0'))
    new_predict= new_predict.withColumn('joint_pred_1',j_predic('nb_pred_1','svm_pred_1'))
    new_predict= new_predict.withColumn('joint_pred_2',j_predic('nb_pred_2','svm_pred_2'))
    
    #do one-hot encoding in new_predict in order to get the new meta_features
    #using the same pipe used for the training data
    new_meta_features= gen_meta_feature_pipeline_model.transform(new_predict)

    #final predictions- logistic regression
    final_predictions= meta_classifier.transform(new_meta_features)
    
    #select asked columns
    final_predictions = final_predictions.select(final_predictions['id'],final_predictions['label'],final_predictions['final_prediction'])
    
    return final_predictions





























